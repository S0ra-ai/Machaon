"""The CLI is a public interface: exit codes and JSON output are tested."""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from machaon.cli import main

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(list(argv))
    return code, out.getvalue(), err.getvalue()


class TestCli(unittest.TestCase):
    def test_validate_accepts_the_example(self):
        code, out, _ = run("validate", os.path.join(ROOT, "examples/edit_spec.json"))
        self.assertEqual(code, 0)
        self.assertIn("rank1_constrained", out)

    def test_validate_reports_malformed_json_without_a_traceback(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            handle.write("{not json")
            path = handle.name
        try:
            code, _, err = run("validate", path)
        finally:
            os.unlink(path)
        self.assertEqual(code, 2)
        self.assertIn("error:", err)
        self.assertNotIn("Traceback", err)

    def test_gate_returns_one_on_a_failing_edit(self):
        code, out, _ = run("gate", os.path.join(ROOT, "examples/metrics_fail.json"))
        self.assertEqual(code, 1)
        self.assertIn("FAIL", out)

    def test_gate_returns_zero_on_a_passing_edit(self):
        code, out, _ = run("gate", os.path.join(ROOT, "examples/metrics_pass.json"))
        self.assertEqual(code, 0)
        self.assertIn("PASS", out)

    def test_wsp_accepts_a_repair_report(self):
        code, out, _ = run("wsp", os.path.join(ROOT, "examples/wsp_report.json"))
        self.assertEqual(code, 0)
        report = json.loads(out)
        self.assertTrue(report["accepted"])
        self.assertEqual(report["label"], "repair")

    def test_wsp_rejects_a_non_specific_report(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump({
                "base": {"rewrite_acc": 0.2},
                "candidate": {"rewrite_acc": 0.9},
                "controls": {"random": {"rewrite_acc": 0.88}},
                "headroom": {"rewrite_acc": 0.5},
            }, handle)
            path = handle.name
        try:
            code, out, _ = run("wsp", path)
        finally:
            os.unlink(path)
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(out)["label"], "non_specific")

    def test_wsp_saves_and_verifies_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = os.path.join(tmp, "bundle")
            metadata = os.path.join(tmp, "metadata.json")
            with open(metadata, "w") as handle:
                json.dump({"model": "toy", "seed": 0}, handle)
            code, out, _ = run(
                "wsp", os.path.join(ROOT, "examples/wsp_report.json"),
                "--save", bundle, "--metadata", metadata,
            )
            self.assertEqual(code, 0)
            report = json.loads(out)
            self.assertEqual(report["label"], "repair")
            self.assertIn("sha256", report["evidence"])

            code, out, _ = run("wsp-verify", bundle)
            self.assertEqual(code, 0)
            verified = json.loads(out)
            self.assertTrue(verified["valid"])
            self.assertEqual(verified["label"], "repair")

            with open(os.path.join(bundle, "evidence.json")) as handle:
                document = json.load(handle)
            document["report"]["label"] = "acquisition"
            with open(os.path.join(bundle, "evidence.json"), "w") as handle:
                json.dump(document, handle)
            code, _, err = run("wsp-verify", bundle)
            self.assertEqual(code, 2)
            self.assertIn("address mismatch", err)

    def test_wsp_benchmark_fixture_matches(self):
        code, out, _ = run(
            "wsp-bench", os.path.join(ROOT, "examples", "wsp_benchmark.json")
        )
        self.assertEqual(code, 0)
        report = json.loads(out)
        self.assertTrue(report["summary"]["all_matched"])
        self.assertEqual(report["summary"]["n_cases"], 2)

    def test_wsp_benchmark_fails_on_an_unexpected_repair(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump({
                "cases": [{
                    "case_id": "expected-but-flat",
                    "failure_type": "obstruction",
                    "expected_repairable": True,
                    "base": {"rewrite_acc": 0.2},
                    "candidate": {"rewrite_acc": 0.2},
                    "controls": {"random": {"rewrite_acc": 0.2}},
                    "headroom": {"rewrite_acc": 0.2},
                }]
            }, handle)
            path = handle.name
        try:
            code, out, _ = run("wsp-bench", path)
        finally:
            os.unlink(path)
        self.assertEqual(code, 1)
        self.assertFalse(json.loads(out)["summary"]["all_matched"])

    def test_wsp_benchmark_can_refuse_synthetic_fixtures(self):
        code, _, err = run(
            "wsp-bench", os.path.join(ROOT, "examples", "wsp_benchmark.json"),
            "--require-measured",
        )
        self.assertEqual(code, 2)
        self.assertIn("synthetic cases", err)

    def test_measured_wsp_benchmark_passes_required_gate(self):
        code, out, _ = run(
            "wsp-bench", os.path.join(ROOT, "examples", "wsp_measured_smoke.json"),
            "--require-measured",
        )
        self.assertEqual(code, 0)
        report = json.loads(out)
        self.assertTrue(report["summary"]["all_matched"])
        self.assertEqual(report["summary"]["source_kinds"], {"measured": 1})

    def test_plan_selects_a_layer_from_site_scores(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump({
                "failure_type": "obstruction",
                "layer_scores": {"12": 0.1, "14": 0.3},
            }, handle)
            path = handle.name
        try:
            code, out, _ = run("plan", path)
        finally:
            os.unlink(path)
        self.assertEqual(code, 0)
        plan = json.loads(out)
        self.assertEqual(plan["route"], "edit")
        self.assertEqual(plan["layer"], 14)

    def test_diagnose_classifies_and_builds_a_plan(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump({
                "base": {"accuracy": 0.2},
                "headroom": {"accuracy": 0.9},
                "mechanism": {
                    "base": {"loop_rate": 0.9},
                    "headroom": {"loop_rate": 0.2},
                },
                "plan": {"layer_scores": {"12": 0.3}},
            }, handle)
            path = handle.name
        try:
            code, out, _ = run("diagnose", path)
        finally:
            os.unlink(path)
        self.assertEqual(code, 0)
        result = json.loads(out)
        self.assertEqual(result["diagnosis"]["failure_type"], "obstruction")
        self.assertEqual(result["plan"]["route"], "edit")

    def test_doctor_json_has_core_and_optional_sections(self):
        code, out, _ = run("doctor", "--json")
        self.assertEqual(code, 0)
        report = json.loads(out)
        self.assertEqual(report["machaon"], __import__("machaon").__version__)
        self.assertTrue(all(report["core"].values()))
        self.assertIn("torch", report["packages"])
        self.assertIn("devices", report)
        self.assertIn("default", report["devices"])
        self.assertIn("cuda", report["devices"])
        self.assertIn("mps", report["devices"])

    def test_web_forwards_launch_controls(self):
        import types

        fake = types.ModuleType("machaon.web.app")
        captured = []

        def web_main(argv):
            captured.append(argv)
            return 0

        fake.main = web_main
        with mock.patch.dict(sys.modules, {"machaon.web.app": fake}):
            code, _, err = run(
                "web", "--allowed-root", "models",
                "--port", "8770", "--no-browser",
            )
        self.assertEqual(code, 0)
        self.assertEqual(err, "")
        self.assertIn("--allowed-root", captured[0])
        self.assertIn("--port", captured[0])
        self.assertIn("--no-browser", captured[0])

    def test_intent_prints_a_reviewable_contract(self):
        code, out, _ = run(
            "intent", "The Eiffel Tower is in the city of -> Rome"
        )
        self.assertEqual(code, 0)
        self.assertIn("kind       : teach", out)
        self.assertIn("confidence : high", out)
        self.assertIn("machaon teach --model", out)

    def test_intent_json_is_machine_readable(self):
        code, out, _ = run(
            "intent", "教会模型：北京的首都是 → 北京", "--json"
        )
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(payload["kind"], "teach")
        self.assertEqual(payload["prompt"], "北京的首都是")
        self.assertEqual(payload["answer"], "北京")
        self.assertIn("command", payload)

    def test_intent_refuses_unstructured_text(self):
        code, _, err = run("intent", "please make it better")
        self.assertEqual(code, 2)
        self.assertIn("error:", err)
        self.assertIn("Try one of", err)

    def test_program_plan_resolves_without_a_model(self):
        code, out, _ = run(
            "program", os.path.join(ROOT, "examples/hello.machaon"), "--plan"
        )
        self.assertEqual(code, 0)
        self.assertIn("weight program", out)
        self.assertIn("The Eiffel Tower is in the city of", out)
        self.assertIn("verify", out.lower())

    def test_program_plan_json_is_reviewable(self):
        code, out, _ = run(
            "program", os.path.join(ROOT, "examples/hello.machaon"),
            "--plan", "--json",
        )
        self.assertEqual(code, 0)
        plan = json.loads(out)
        self.assertEqual(plan["format"], "machaon.program/1")
        self.assertEqual(len(plan["facts"]), 2)
        self.assertEqual(plan["facts"][0]["prompt"],
                         "The Eiffel Tower is in the city of")
        self.assertIn("command", plan)
        self.assertIn("preflight", plan)
        self.assertFalse(plan["preflight"]["model_exists"])
        self.assertEqual(plan["preflight"]["fetch"]["starter"], "smol135m")

    def test_program_plan_refuses_run_only_flags(self):
        code, _, err = run(
            "program", os.path.join(ROOT, "examples/hello.machaon"),
            "--plan", "--no-verify",
        )
        self.assertEqual(code, 2)
        self.assertIn("only be used when running", err)

    def test_program_plan_names_a_missing_model_without_fetch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "local.machaon"
            path.write_text(
                "model: ./missing\noutput: ./out\n"
                "teach: The capital is -> Paris\n",
                encoding="utf-8",
            )
            code, out, _ = run("program", str(path), "--plan")
        self.assertEqual(code, 0)
        self.assertIn("download it first", out)

    def test_program_reports_parse_errors_without_a_traceback(self):
        with tempfile.NamedTemporaryFile("w", suffix=".machaon", delete=False) as handle:
            handle.write("model: base\noutput: out\nmagic: yes\n")
            path = handle.name
        try:
            code, _, err = run("program", path, "--plan")
        finally:
            os.unlink(path)
        self.assertEqual(code, 2)
        self.assertIn("unknown field", err)
        self.assertNotIn("Traceback", err)

    def test_program_prints_rejection_checks(self):
        class FakeRun:
            status = "stopped"
            accepted = False
            summary = "rejected"
            model_fetched = False
            verification = None
            report_path = None
            report_sha256 = None
            run = {
                "applied": 0,
                "rejected": 1,
                "already_true": 0,
                "reason": "fact 1 was rejected",
                "steps": [{
                    "index": 1,
                    "status": "rejected",
                    "result": {
                        "checks": [{
                            "name": "panel_top1",
                            "passed": False,
                            "requirement": ">= 0.875 of the panel",
                            "value": 0.5,
                        }]
                    },
                }],
            }

            def to_dict(self):
                return {"run": self.run}

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lesson.machaon"
            path.write_text(
                "model: ./base\noutput: ./edited\n"
                "teach: The capital is -> Paris\n",
                encoding="utf-8",
            )
            with mock.patch(
                "machaon.program.run_program", return_value=FakeRun()
            ):
                code, out, _ = run("program", str(path))
        self.assertEqual(code, 1)
        self.assertIn("failure    : fact 1 REJECTED", out)
        self.assertIn("panel_top1", out)
        self.assertIn("[0.5]", out)

    def test_program_report_verification(self):
        from machaon.program import parse_program, write_program_report

        with tempfile.TemporaryDirectory() as tmp:
            program = parse_program(
                "model: ./base\noutput: ./out\n"
                "teach: The capital of France is -> Paris\n",
                source="lesson.machaon",
                base_dir=tmp,
            )
            report_path = os.path.join(tmp, "program-report.json")
            write_program_report(report_path, {
                "format": "machaon.program/1",
                "program": program.to_dict(),
                "run": {"status": "no_edits", "final": None},
                "verification": {"passed": True, "facts": []},
                "status": "no_edits",
                "accepted": True,
                "summary": "already true",
            })
            code, out, _ = run("program-verify", report_path)
            self.assertEqual(code, 0)
            self.assertIn("integrity  : VALID", out)
            self.assertIn("NO_EDITS", out)

            code, out, _ = run("program-verify", report_path, "--json")
            self.assertEqual(code, 0)
            payload = json.loads(out)
            self.assertTrue(payload["valid"])
            self.assertEqual(len(payload["report_sha256"]), 64)

            valid_text = Path(report_path).read_text(encoding="utf-8")
            with mock.patch("sys.stdin", io.StringIO(valid_text)):
                code, out, _ = run("program-verify", "-")
            self.assertEqual(code, 0)
            self.assertIn("integrity  : VALID", out)

            with open(report_path) as handle:
                tampered = json.load(handle)
            tampered["accepted"] = False
            with open(report_path, "w") as handle:
                json.dump(tampered, handle)
            code, _, err = run("program-verify", report_path)
            self.assertEqual(code, 2)
            self.assertIn("address mismatch", err)

            code, out, _ = run("program-verify", report_path, "--json")
            self.assertEqual(code, 2)
            payload = json.loads(out)
            self.assertFalse(payload["valid"])
            self.assertIn("address mismatch", payload["error"])

    def test_program_new_writes_a_runnable_program(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lesson.machaon")
            code, out, _ = run(
                "program-new", path,
                "--teach", "The Eiffel Tower is in the city of -> Rome",
                "--protect", "Two plus two equals",
                "--budget", "0.12",
            )
            self.assertEqual(code, 0)
            self.assertIn("created", out)
            text = Path(path).read_text(encoding="utf-8")
            self.assertIn("fetch: smol135m", text)
            self.assertIn("budget: 0.12", text)
            self.assertIn("teach: The Eiffel Tower", text)

            code, out, _ = run("program", path, "--plan", "--json")
            self.assertEqual(code, 0)
            plan = json.loads(out)
            self.assertEqual(plan["fetch"], "smol135m")
            self.assertEqual(plan["protect"], ["Two plus two equals"])

            code, _, err = run(
                "program-new", path,
                "--teach", "The capital is -> Paris",
            )
            self.assertEqual(code, 2)
            self.assertIn("already exists", err)
            code, _, _ = run(
                "program-new", path, "--force",
                "--teach", "The capital is -> Paris",
            )
            self.assertEqual(code, 0)
            self.assertIn("The capital is", Path(path).read_text(encoding="utf-8"))

    def test_program_new_stdout_and_custom_model(self):
        code, out, _ = run(
            "program-new", "-",
            "--model", "./models/custom",
            "--teach", "The capital is -> Paris",
        )
        self.assertEqual(code, 0)
        self.assertIn("model: ./models/custom", out)
        self.assertIn("output: ./models/custom-machaon", out)
        self.assertNotIn("fetch:", out)

    def test_program_new_uses_the_requested_starter_name(self):
        code, out, _ = run(
            "program-new", "-", "--fetch", "qwen05b",
            "--teach", "北京的首都是 -> 北京",
        )
        self.assertEqual(code, 0)
        self.assertIn("model: ./models/Qwen2.5-0.5B", out)
        self.assertIn("fetch: qwen05b", out)

    def test_program_new_reads_fact_files_and_advanced_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            facts = Path(tmp) / "facts.txt"
            facts.write_text(
                "# facts for the generated program\n"
                "The capital of France is -> Paris\n"
                "\n"
                "The city is -> Rome\n",
                encoding="utf-8",
            )
            protect = Path(tmp) / "protect.txt"
            protect.write_text(
                "Two plus two equals\nWater freezes at a temperature of\n",
                encoding="utf-8",
            )
            path = Path(tmp) / "advanced.machaon"
            code, _, _ = run(
                "program-new", str(path),
                "--facts", str(facts),
                "--protect-file", str(protect),
                "--layers", "1-2",
                "--solver", "gradient",
                "--steps", "10",
                "--lr", "0.1",
                "--max-layers", "2",
                "--max-kl", "0.02",
                "--min-panel-agreement", "0.5",
                "--dtype", "auto",
                "--quantize", "int8",
                "--device", "cpu",
                "--no-verify",
                "--continue-on-reject",
                "--keep-steps",
            )
            self.assertEqual(code, 0)
            code, out, _ = run("program", str(path), "--plan", "--json")
        self.assertEqual(code, 0)
        plan = json.loads(out)
        self.assertEqual(len(plan["facts"]), 2)
        self.assertEqual(plan["protect"], [
            "Two plus two equals",
            "Water freezes at a temperature of",
        ])
        self.assertEqual(plan["runtime"]["layers"], [1, 2])
        self.assertEqual(plan["runtime"]["solver"], "gradient")
        self.assertEqual(plan["runtime"]["steps"], 10)
        self.assertEqual(plan["runtime"]["max_layers"], 2)
        self.assertEqual(plan["runtime"]["dtype"], "auto")
        self.assertEqual(plan["runtime"]["quantize"], "int8")
        self.assertEqual(plan["runtime"]["device"], "cpu")
        self.assertFalse(plan["verify"])
        self.assertTrue(plan["continue_on_reject"])
        self.assertTrue(plan["keep_steps"])
        self.assertEqual(plan["policy"]["max_kl_p95"], 0.02)
        self.assertEqual(plan["policy"]["min_panel_agreement"], 0.5)

    def test_program_new_requires_a_fact_source(self):
        code, _, err = run("program-new", "-")
        self.assertEqual(code, 2)
        self.assertIn("--teach or --facts", err)

    def test_program_new_refuses_conflicting_fetch_flags(self):
        code, _, err = run(
            "program-new", "-", "--fetch", "gpt2", "--no-fetch",
            "--teach", "The capital is -> Paris",
        )
        self.assertEqual(code, 2)
        self.assertIn("not both", err)

    def test_teach_refuses_conflicting_inputs_before_loading(self):
        code, _, err = run(
            "teach", "--model", "/tmp/none",
            "--say", "The capital is -> Paris", "--prompt", "x",
        )
        self.assertEqual(code, 2)
        self.assertIn("not both", err)

    def test_teach_refuses_dry_run_with_output(self):
        code, _, err = run(
            "teach", "--model", "/tmp/none",
            "--say", "The capital is -> Paris", "--dry-run", "--out", "/tmp/out",
        )
        self.assertEqual(code, 2)
        self.assertIn("--dry-run", err)

    def test_teach_refuses_no_save_with_output(self):
        code, _, err = run(
            "teach", "--model", "/tmp/none",
            "--say", "The capital is -> Paris", "--no-save", "--out", "/tmp/out",
        )
        self.assertEqual(code, 2)
        self.assertIn("--no-save", err)

    def test_teach_needs_prompt_and_answer_without_say(self):
        code, _, err = run("teach", "--model", "/tmp/none", "--prompt", "x")
        self.assertEqual(code, 2)
        self.assertIn("--say", err)

    def test_teach_refuses_a_missing_checkpoint_with_a_download_hint(self):
        code, _, err = run(
            "teach", "--model", "/tmp/definitely-not-a-checkpoint",
            "--say", "The capital is -> Paris",
        )
        self.assertEqual(code, 2)
        self.assertIn("no such checkpoint", err)
        self.assertIn("fetch_model.py", err)

    def test_teach_refuses_existing_artifact_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            evidence = os.path.join(tmp, "evidence")
            os.makedirs(evidence)
            code, _, err = run(
                "teach", "--model", tmp,
                "--say", "The capital is -> Paris",
                "--evidence-out", evidence,
            )
            self.assertEqual(code, 2)
            self.assertIn("already exists", err)

    def test_teach_needs_a_directory_not_a_single_weight_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".safetensors", delete=False) as handle:
            handle.write("{}")
            path = handle.name
        try:
            code, _, err = run(
                "teach", "--model", path, "--say", "The capital is -> Paris"
            )
        finally:
            os.unlink(path)
        self.assertEqual(code, 2)
        self.assertIn("checkpoint directory", err)

    def test_layer_ranges_are_expanded(self):
        from machaon.cli import _parse_layers

        self.assertEqual(_parse_layers("8,10,12"), [8, 10, 12])
        self.assertEqual(_parse_layers("8-10"), [8, 9, 10])
        self.assertEqual(_parse_layers("3, 2-4"), [2, 3, 4])
        with self.assertRaises(ValueError):
            _parse_layers("9-7")

    def test_default_output_is_a_sibling_directory(self):
        from machaon.cli import _default_out

        self.assertTrue(_default_out("/models/qwen").endswith("qwen-machaon"))
        self.assertTrue(
            _default_out("/models/model.safetensors").endswith("model-machaon")
        )

    def test_default_output_leaves_the_huggingface_cache(self):
        from machaon.cli import _default_out

        path = (
            "/cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/"
            "snapshots/abc123"
        )
        output = _default_out(path)
        self.assertTrue(output.endswith("Qwen2.5-0.5B-machaon"))
        self.assertNotIn("snapshots", output)

    def test_discard_artifacts_removes_partial_outputs(self):
        from machaon.cli import _discard_artifacts

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = root / "evidence"
            edit = root / "edit"
            report = root / "report.json"
            evidence.mkdir()
            edit.mkdir()
            report.write_text("{}")
            _discard_artifacts({
                "evidence": {"path": str(evidence)},
                "edit": {"path": str(edit)},
                "report": str(report),
            })
            self.assertFalse(evidence.exists())
            self.assertFalse(edit.exists())
            self.assertFalse(report.exists())

    def test_fetch_lists_the_starter_catalogue(self):
        code, out, _ = run("fetch", "--list")
        self.assertEqual(code, 0)
        self.assertIn("qwen05b", out)
        self.assertIn("Qwen/Qwen2.5-0.5B", out)

    def test_fetch_without_a_source_returns_two(self):
        code, _, err = run("fetch")
        self.assertEqual(code, 2)
        self.assertIn("--repo", err)

    def test_teach_many_refuses_a_missing_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            facts = Path(tmp) / "facts.txt"
            facts.write_text("The capital is -> Paris\n")
            code, _, err = run(
                "teach-many", "--model", "/tmp/definitely-not-a-model",
                "--facts", str(facts),
            )
        self.assertEqual(code, 2)
        self.assertIn("no such checkpoint", err)

    def test_teach_many_refuses_an_empty_facts_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            facts = Path(tmp) / "facts.txt"
            facts.write_text("# no facts here\n")
            code, _, err = run(
                "teach-many", "--model", tmp, "--facts", str(facts)
            )
        self.assertEqual(code, 2)
        self.assertIn("contains no facts", err)

    def test_teach_many_refuses_a_missing_facts_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, _, err = run(
                "teach-many", "--model", tmp,
                "--facts", str(Path(tmp) / "missing.txt"),
            )
        self.assertEqual(code, 2)
        self.assertIn("error:", err)

    def test_first_refuses_an_unstructured_sentence_before_fetching(self):
        code, _, err = run(
            "first", "--model", "/tmp/definitely-not-a-model",
            "--say", "please make it better",
        )
        self.assertEqual(code, 2)
        self.assertIn("could not read", err)

    def test_first_explains_a_missing_real_model_runtime(self):
        with mock.patch(
            "machaon.first.missing_runtime_packages",
            return_value=("torch", "transformers"),
        ):
            code, out, err = run(
                "first", "--model", "/tmp/definitely-not-a-model",
                "--say", "The capital is -> Paris",
            )
        self.assertEqual(code, 2)
        self.assertIn("machaon[hf]", err)
        self.assertIn("torch", err)
        self.assertNotIn("Traceback", err + out)

    def test_first_rejects_no_save_with_out(self):
        code, _, err = run(
            "first", "--model", "/tmp/none", "--out", "/tmp/out", "--no-save"
        )
        self.assertEqual(code, 2)
        self.assertIn("--no-save", err)

    def test_first_writes_a_matching_program_before_the_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            program_path = Path(tmp) / "first.machaon"
            output_path = Path(tmp) / "edited"

            class FakeResult:
                accepted = True

                def to_dict(self):
                    return {"status": "applied", "accepted": True}

            with mock.patch(
                "machaon.first.first_edit", return_value=FakeResult()
            ):
                code, out, _ = run(
                    "first",
                    "--model", str(Path(tmp) / "SmolLM2-135M"),
                    "--out", str(output_path),
                    "--program-out", str(program_path),
                    "--say", "The capital of France is -> Paris",
                    "--json",
                )
            self.assertEqual(code, 0)
            text = program_path.read_text(encoding="utf-8")
            self.assertIn("model: SmolLM2-135M", text)
            self.assertIn("fetch: smol135m", text)
            self.assertIn("output: edited", text)
            self.assertIn("teach: The capital of France is -> Paris", text)
            payload = json.loads(out)
            self.assertEqual(payload["program_path"], str(program_path))
            self.assertEqual(len(payload["program_sha256"]), 64)

            code, _, err = run(
                "first",
                "--model", str(Path(tmp) / "SmolLM2-135M"),
                "--out", str(output_path),
                "--program-out", str(program_path),
                "--say", "The capital of France is -> Paris",
                "--json",
            )
        self.assertEqual(code, 2)
        self.assertIn("--program-force", err)

    def test_first_refuses_program_inside_the_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "edited"
            code, _, err = run(
                "first",
                "--model", str(Path(tmp) / "SmolLM2-135M"),
                "--out", str(output),
                "--program-out", str(output / "first.machaon"),
                "--say", "The capital of France is -> Paris",
            )
        self.assertEqual(code, 2)
        self.assertIn("inside the output directory", err)

    def test_check_reports_pass_fail_and_json_without_a_real_model(self):
        class FakeSession:
            def __init__(self, accuracy):
                self.accuracy = accuracy

            def evaluate(self, prompts, targets):
                return {"accuracy": self.accuracy, "n": len(prompts)}

        with mock.patch(
            "machaon.session.Session.open",
            return_value=FakeSession(1.0),
        ):
            code, out, _ = run(
                "check", "--model", ROOT,
                "--say", "The Eiffel Tower is in the city of -> Rome",
            )
        self.assertEqual(code, 0)
        self.assertIn("PASS", out)
        self.assertIn(" Rome", out)

        with mock.patch(
            "machaon.session.Session.open",
            return_value=FakeSession(0.0),
        ):
            code, out, _ = run(
                "check", "--model", ROOT,
                "--say", "The Eiffel Tower is in the city of -> Rome",
                "--json",
            )
        self.assertEqual(code, 1)
        payload = json.loads(out)
        self.assertEqual(payload["status"], "fail")
        self.assertFalse(payload["passed"])
        self.assertEqual(payload["answer_text"], " Rome")

    def test_check_refuses_quantize_without_auto_dtype(self):
        code, _, err = run(
            "check", "--model", ROOT,
            "--say", "The Eiffel Tower is in the city of -> Rome",
            "--dtype", "float32", "--quantize", "int4",
        )
        self.assertEqual(code, 2)
        self.assertIn("--quantize", err)

    def test_check_facts_reuses_one_saved_checkpoint(self):
        class FakeSession:
            def evaluate(self, prompts, targets):
                return {"accuracy": 1.0, "n": len(prompts)}

        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
            handle.write("The capital is -> Paris\n")
            handle.write("The city is -> Rome\n")
            path = handle.name
        try:
            with mock.patch(
                "machaon.session.Session.open",
                return_value=FakeSession(),
            ):
                code, out, _ = run(
                    "check", "--model", ROOT, "--facts", path, "--json"
                )
        finally:
            os.unlink(path)
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertTrue(payload["passed"])
        self.assertEqual(len(payload["facts"]), 2)
        self.assertEqual(payload["facts"][0]["answer_text"], " Paris")

    def test_start_prints_the_three_entry_points(self):
        code, out, _ = run("start")
        self.assertEqual(code, 0)
        self.assertIn("machaon first", out)
        self.assertIn("machaon demo", out)
        self.assertIn("machaon teach", out)
        self.assertIn("machaon check", out)
        self.assertIn("machaon-web", out)

    def test_module_entry_point(self):
        result = subprocess.run(
            [sys.executable, "-m", "machaon", "--version"],
            cwd=ROOT, capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(result.stdout.startswith("machaon "))

    def test_unknown_input_has_a_nonzero_exit(self):
        code, _, err = run("gate", os.path.join(ROOT, "does-not-exist.json"))
        self.assertEqual(code, 2)
        self.assertIn("error:", err)


if __name__ == "__main__":
    unittest.main()
