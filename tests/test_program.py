"""Portable programs compose teach-many, save and a reload check."""
import json
import tempfile
import unittest
from pathlib import Path

from machaon.program import (
    ProgramError,
    ProgramReportError,
    load_program,
    parse_layers,
    parse_program,
    read_program_report,
    render_program_text,
    run_program,
    verify_program_report,
    write_program_report,
)

TEXT_PROGRAM = """\
# a portable lesson
version: 1
name: First lesson
model: ./models/base
fetch: smol135m
output: ./models/lesson
budget: 0.10
max_rel_norm: 0.02
layers: 18-20
verify: yes

teach: The Eiffel Tower is in the city of -> Rome
teach: 北京的首都是 -> 北京

protect: The capital of Germany is
protect: Water freezes at a temperature of
protect: The capital of Germany is
"""


def fake_result(status="applied", final=True):
    report = {
        "status": status,
        "final": "/tmp/out/final" if final else None,
        "out_dir": "/tmp/out",
        "facts": 1,
        "steps": [{"status": status}],
        "summary": "finished",
    }

    class Result:
        def to_dict(self):
            return dict(report)

    return Result()


def fake_check(passed=True):
    calls = []

    def check(model, facts, **kwargs):
        calls.append((model, facts, kwargs))
        return {
            "passed": passed,
            "model": model,
            "facts": [
                {
                    "index": index,
                    "prompt": prompt,
                    "answer": answer,
                    "answer_text": answer,
                    "passed": passed,
                }
                for index, (prompt, answer) in enumerate(facts, 1)
            ],
        }

    return check, calls


class TestProgramParser(unittest.TestCase):
    def test_text_program_resolves_paths_and_compiles_facts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "lesson.machaon"
            path.write_text(TEXT_PROGRAM, encoding="utf-8")
            program = load_program(path)

        self.assertEqual(program.model, str((root / "models/base").resolve()))
        self.assertEqual(program.output, str((root / "models/lesson").resolve()))
        self.assertEqual(program.fetch, "smol135m")
        self.assertEqual(program.budget, 0.10)
        self.assertEqual(program.layers, (18, 19, 20))
        self.assertEqual(program.name, "First lesson")
        self.assertEqual(len(program.facts), 2)
        self.assertEqual(program.facts[0]["prompt"], "The Eiffel Tower is in the city of")
        self.assertEqual(program.facts[0]["answer"], "Rome")
        self.assertEqual(program.facts[1]["prompt"], "北京的首都是")
        self.assertEqual(program.protect, (
            "The capital of Germany is",
            "Water freezes at a temperature of",
        ))
        self.assertTrue(program.verify)
        self.assertEqual(len(program.sha256), 64)

    def test_json_program_uses_the_same_contract(self):
        payload = {
            "version": 1,
            "model": "base",
            "output": "edited",
            "facts": [
                "The capital of France is -> Paris",
                {"prompt": "The river is", "answer": " the Seine"},
            ],
            "protect": ["Two plus two equals"],
            "layers": [4, 6],
            "verify": False,
            "dtype": "float16",
        }
        program = parse_program(
            json.dumps(payload), source="inline.json", base_dir="."
        )
        self.assertEqual(program.facts[1]["prompt"], "The river is")
        self.assertEqual(program.facts[1]["answer"], " the Seine")
        self.assertEqual(program.layers, (4, 6))
        self.assertFalse(program.verify)
        self.assertEqual(program.source_format, "json")

    def test_rendered_program_round_trips_through_the_parser(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = render_program_text(
                model="./models/SmolLM2-135M",
                output="./models/lesson",
                fetch="smol135m",
                name="A generated lesson",
                teach=[
                    "The Eiffel Tower is in the city of -> Rome",
                    "The capital of France is -> Paris",
                ],
                protect=["Two plus two equals"],
                budget=0.12,
                max_rel_norm=0.03,
            )
            path = Path(tmp) / "generated.machaon"
            path.write_text(text, encoding="utf-8")
            program = load_program(path)
        self.assertEqual(program.name, "A generated lesson")
        self.assertEqual(program.fetch, "smol135m")
        self.assertEqual(program.budget, 0.12)
        self.assertEqual(program.max_rel_norm, 0.03)
        self.assertEqual(len(program.facts), 2)
        self.assertEqual(program.protect, ("Two plus two equals",))
        self.assertTrue(program.verify)
        self.assertIn("program-new", text)

    def test_rendered_program_refuses_multiline_values(self):
        with self.assertRaisesRegex(ProgramError, "single line"):
            render_program_text(
                model="./base",
                output="./edited",
                teach=["The capital is -> Paris\nunexpected"],
            )

    def test_rendered_program_can_emit_advanced_fields(self):
        text = render_program_text(
            model="./base",
            output="./edited",
            teach=["The capital is -> Paris"],
            layers=[4, 5],
            solver="gradient",
            dtype="auto",
            quantize="int8",
            min_panel_agreement=1.0,
            continue_on_reject=True,
            keep_steps=True,
        )
        program = parse_program(text, source="inline.machaon", base_dir=".")
        self.assertEqual(program.layers, (4, 5))
        self.assertEqual(program.solver, "gradient")
        self.assertEqual(program.dtype, "auto")
        self.assertEqual(program.quantize, "int8")
        self.assertEqual(program.min_panel_agreement, 1.0)
        self.assertTrue(program.continue_on_reject)
        self.assertTrue(program.keep_steps)

    def test_unknown_field_names_the_line(self):
        with self.assertRaisesRegex(ProgramError, ":2: unknown field"):
            parse_program(
                "model: base\nmagic: yes\noutput: out\nteach: a -> b\n",
                source="bad.machaon",
                base_dir=".",
            )

    def test_duplicate_single_field_is_refused(self):
        with self.assertRaisesRegex(ProgramError, "already set"):
            parse_program(
                "model: base\nmodel: other\noutput: out\nteach: a -> b\n",
                source="bad.machaon",
                base_dir=".",
            )

    def test_bad_teaching_sentence_names_the_line(self):
        with self.assertRaisesRegex(ProgramError, "teach entry 1"):
            parse_program(
                "model: base\noutput: out\nteach: please improve this\n",
                source="bad.machaon",
                base_dir=".",
            )

    def test_program_requires_a_model_fact_and_output(self):
        for text, message in (
            ("output: out\nteach: a -> b\n", "model is required"),
            ("model: base\nteach: a -> b\n", "output is required"),
            ("model: base\noutput: out\n", "teach entry"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ProgramError, message):
                    parse_program(text, source="bad.machaon", base_dir=".")

    def test_quantize_requires_auto_dtype(self):
        with self.assertRaisesRegex(ProgramError, "quantize requires"):
            parse_program(
                "model: base\noutput: out\nteach: a -> b\n"
                "dtype: float16\nquantize: int8\n",
                source="bad.machaon",
                base_dir=".",
            )

    def test_unknown_fetch_starter_is_refused_before_a_run(self):
        with self.assertRaisesRegex(ProgramError, "unknown fetch starter"):
            parse_program(
                "model: base\nfetch: giant\noutput: out\nteach: a -> b\n",
                source="bad.machaon",
                base_dir=".",
            )

    def test_layers_accept_ranges_and_lists(self):
        self.assertEqual(parse_layers("2, 4-6"), (2, 4, 5, 6))
        self.assertEqual(parse_layers([3, 1]), (1, 3))
        with self.assertRaises(ProgramError):
            parse_layers("6-4")


class TestRunProgram(unittest.TestCase):
    def _program(self, text=TEXT_PROGRAM):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        (root / "models" / "base").mkdir(parents=True)
        path = root / "lesson.machaon"
        path.write_text(text, encoding="utf-8")
        return root, load_program(path)

    def test_run_saves_and_verifies_saved_bytes(self):
        root, program = self._program()
        check, checks = fake_check()
        calls = []

        def teach(factory, model, facts, out, **kwargs):
            calls.append((model, facts, kwargs))
            out.mkdir(parents=True, exist_ok=True)
            (out / "report.json").write_text("{}")
            return {
                "status": "applied",
                "final": str(out / "final"),
                "out_dir": str(out),
                "facts": len(facts),
                "steps": [{"status": "applied"}, {"status": "already_true"}],
                "summary": "finished",
            }

        result = run_program(
            program,
            teach_many_fn=teach,
            check_fn=check,
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.status, "applied")
        self.assertEqual(result.verification["passed"], True)
        self.assertEqual(len(checks), 1)
        self.assertEqual(checks[0][0], str((root / "models/lesson/final").resolve()))
        self.assertEqual(len(checks[0][1]), 2)
        self.assertEqual(calls[0][2]["policy"].max_rel_norm, 0.02)
        self.assertEqual(calls[0][2]["preserve_prompts"], list(program.protect))
        report = json.loads(
            (root / "models/lesson/program-report.json").read_text()
        )
        self.assertEqual(report["program"]["sha256"], program.sha256)
        self.assertEqual(report["verification"]["passed"], True)
        self.assertEqual(
            report["report_format"], "machaon.program.report/1"
        )
        self.assertEqual(len(report["report_sha256"]), 64)
        self.assertEqual(result.report_sha256, report["report_sha256"])

    def test_verification_failure_is_not_accepted(self):
        root, program = self._program()
        check, _checks = fake_check(passed=False)

        def teach(factory, model, facts, out, **kwargs):
            out.mkdir(parents=True, exist_ok=True)
            return {
                "status": "applied",
                "final": str(out / "final"),
                "out_dir": str(out),
                "facts": len(facts),
                "steps": [{"status": "applied"}],
                "summary": "finished",
            }

        result = run_program(program, teach_many_fn=teach, check_fn=check)
        self.assertFalse(result.accepted)
        self.assertEqual(result.status, "verification_failed")
        self.assertIn("failed verification", result.summary)
        self.assertTrue((root / "models/lesson/program-report.json").exists())

    def test_no_edits_checks_the_base_checkpoint(self):
        root, program = self._program()
        check, checks = fake_check()

        def teach(factory, model, facts, out, **kwargs):
            out.mkdir(parents=True, exist_ok=True)
            return fake_result(status="no_edits", final=False)

        result = run_program(program, teach_many_fn=teach, check_fn=check)
        self.assertTrue(result.accepted)
        self.assertEqual(result.status, "no_edits")
        self.assertEqual(checks[0][0], str((root / "models/base").resolve()))

    def test_no_verify_keeps_the_chain_but_skips_reload(self):
        _root, program = self._program()

        def teach(factory, model, facts, out, **kwargs):
            out.mkdir(parents=True, exist_ok=True)
            return fake_result()

        result = run_program(
            program,
            teach_many_fn=teach,
            check_fn=lambda *a, **k: self.fail("check must not run"),
            verify=False,
        )
        self.assertTrue(result.accepted)
        self.assertIsNone(result.verification)

    def test_missing_model_fetches_the_declared_starter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "lesson.machaon"
            path.write_text(
                "model: ./base\nfetch: smol135m\noutput: ./out\n"
                "teach: The capital is -> Paris\n",
                encoding="utf-8",
            )
            program = load_program(path)
            fetched = []

            def fetch(repo, dest, **kwargs):
                fetched.append((repo, str(dest)))
                Path(dest).mkdir(parents=True)

            def teach(factory, model, facts, out, **kwargs):
                out.mkdir(parents=True, exist_ok=True)
                return fake_result(status="no_edits", final=False)

            result = run_program(
                program,
                fetch_fn=fetch,
                teach_many_fn=teach,
                check_fn=fake_check()[0],
                verify=False,
            )
        self.assertTrue(result.model_fetched)
        self.assertTrue(fetched)
        self.assertTrue(result.accepted)

    def test_missing_model_without_fetch_refuses_before_teaching(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lesson.machaon"
            path.write_text(
                "model: ./base\noutput: ./out\n"
                "teach: The capital is -> Paris\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ProgramError, "fetch: smol135m"):
                run_program(
                    load_program(path),
                    teach_many_fn=lambda *a, **k: self.fail("must not run"),
                )

    def test_nonempty_output_is_refused(self):
        root, program = self._program()
        output = root / "models" / "lesson"
        output.mkdir(parents=True)
        (output / "keep.txt").write_text("do not overwrite")
        with self.assertRaises(FileExistsError):
            run_program(
                program,
                teach_many_fn=lambda *a, **k: self.fail("must not run"),
            )

    def test_output_override_is_used_in_the_report(self):
        root, program = self._program()
        other = root / "other"

        def teach(factory, model, facts, out, **kwargs):
            out.mkdir(parents=True, exist_ok=True)
            return {
                "status": "applied",
                "final": str(out / "final"),
                "out_dir": str(out),
                "facts": len(facts),
                "steps": [{"status": "applied"}],
                "summary": "finished",
            }

        result = run_program(
            program,
            out_override=other,
            teach_many_fn=teach,
            check_fn=fake_check()[0],
        )
        self.assertTrue(result.accepted)
        self.assertTrue((other / "program-report.json").exists())
        self.assertIn("effective_output", result.to_dict())

    def test_report_can_be_written_to_an_explicit_path(self):
        root, program = self._program()
        report_path = root / "artifacts" / "lesson-report.json"

        def teach(factory, model, facts, out, **kwargs):
            out.mkdir(parents=True, exist_ok=True)
            return {
                "status": "applied",
                "final": str(out / "final"),
                "out_dir": str(out),
                "facts": len(facts),
                "steps": [{"status": "applied"}],
                "summary": "finished",
            }

        result = run_program(
            program,
            teach_many_fn=teach,
            check_fn=fake_check()[0],
            report_out=report_path,
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.report_path, str(report_path.resolve()))
        self.assertTrue(report_path.exists())
        report = json.loads(report_path.read_text())
        self.assertEqual(report["program"]["sha256"], program.sha256)

    def test_teaching_error_still_writes_a_report(self):
        root, program = self._program()

        def teach(factory, model, facts, out, **kwargs):
            raise RuntimeError("solver failed")

        result = run_program(program, teach_many_fn=teach)
        self.assertFalse(result.accepted)
        self.assertEqual(result.status, "error")
        self.assertIn("solver failed", result.summary)
        self.assertTrue((root / "models/lesson/program-report.json").exists())

    def test_output_inside_the_model_is_refused(self):
        root, _program = self._program(
            "model: ./models/base\noutput: ./models/base/edited\n"
            "teach: The capital is -> Paris\n"
        )
        with self.assertRaisesRegex(ProgramError, "inside the model"):
            run_program(
                load_program(root / "lesson.machaon"),
                teach_many_fn=lambda *a, **k: self.fail("must not run"),
            )

    def test_verification_exception_is_a_failure(self):
        _root, program = self._program()

        def teach(factory, model, facts, out, **kwargs):
            out.mkdir(parents=True, exist_ok=True)
            return {
                "status": "applied",
                "final": str(out / "final"),
                "out_dir": str(out),
                "facts": len(facts),
                "steps": [{"status": "applied"}],
                "summary": "finished",
            }

        def check(*args, **kwargs):
            raise RuntimeError("reload failed")

        result = run_program(program, teach_many_fn=teach, check_fn=check)
        self.assertFalse(result.accepted)
        self.assertEqual(result.status, "verification_failed")
        self.assertIn("reload failed", result.verification["error"])


class TestProgramReportVerification(unittest.TestCase):
    def _report(self, *, passed=True, accepted=True):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        (root / "models" / "base").mkdir(parents=True)
        program_path = root / "lesson.machaon"
        program_path.write_text(TEXT_PROGRAM, encoding="utf-8")
        program = load_program(program_path)
        payload = {
            "format": "machaon.program/1",
            "program": program.to_dict(),
            "run": {
                "status": "applied",
                "final": str(root / "models" / "lesson" / "final"),
                "out_dir": str(root / "models" / "lesson"),
                "facts": len(program.facts),
            },
            "verification": {
                "passed": passed,
                "facts": [
                    {
                        "index": index,
                        "prompt": fact["prompt"],
                        "answer": fact["answer"],
                        "passed": passed,
                    }
                    for index, fact in enumerate(program.facts, 1)
                ],
            },
            "status": "applied" if accepted else "verification_failed",
            "accepted": accepted,
            "summary": "a recorded run",
            "model_fetched": False,
            "effective_output": program.output,
        }
        report_path = root / "program-report.json"
        document = write_program_report(report_path, payload)
        return root, program_path, report_path, document

    def test_report_round_trips_with_a_content_address(self):
        _root, _program_path, path, document = self._report()
        loaded = read_program_report(path)
        self.assertEqual(loaded, document)
        self.assertEqual(document["report_format_version"], 1)
        self.assertEqual(
            document["report_sha256"], read_program_report(path)["report_sha256"]
        )

    def test_modified_report_is_refused(self):
        _root, _program_path, path, document = self._report()
        document["accepted"] = False
        path.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(ProgramReportError, "address mismatch"):
            read_program_report(path)

    def test_program_hash_cross_check(self):
        root, program_path, path, _document = self._report()
        result = verify_program_report(path, program_path=program_path)
        self.assertTrue(result["valid"])
        self.assertTrue(result["program_hash_matches"])
        self.assertIsNone(result["model_check"])

        other = root / "other.machaon"
        other.write_text(
            TEXT_PROGRAM.replace("First lesson", "Another lesson"),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ProgramReportError, "program hash"):
            verify_program_report(path, program_path=other)

    def test_optional_model_check_reloads_the_checkpoint(self):
        _root, _program_path, path, _document = self._report()
        check, calls = fake_check()
        result = verify_program_report(path, check_model=True, check_fn=check)
        self.assertTrue(result["model_check"]["performed"])
        self.assertTrue(result["model_check"]["passed"])
        self.assertTrue(result["model_check"]["matches_report"])
        self.assertEqual(len(calls[0][1]), 2)
        self.assertTrue(calls[0][0].endswith("/models/lesson/final"))

    def test_model_check_can_disagree_with_the_recorded_claim(self):
        _root, _program_path, path, _document = self._report()
        check, _calls = fake_check(passed=False)
        result = verify_program_report(path, check_model=True, check_fn=check)
        self.assertFalse(result["model_check"]["passed"])
        self.assertFalse(result["model_check"]["matches_report"])

    def test_missing_saved_checkpoint_is_reported(self):
        _root, _program_path, path, document = self._report(accepted=False)
        document["run"]["final"] = None
        write_program_report(path, document)
        with self.assertRaisesRegex(ProgramReportError, "no saved checkpoint"):
            verify_program_report(path, check_model=True, check_fn=fake_check()[0])

    def test_old_report_without_an_address_is_refused(self):
        _root, _program_path, path, document = self._report()
        document.pop("report_format")
        path.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(ProgramReportError, "unsupported report format"):
            read_program_report(path)


if __name__ == "__main__":
    unittest.main()
