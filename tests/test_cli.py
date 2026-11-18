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

    def test_start_prints_the_three_entry_points(self):
        code, out, _ = run("start")
        self.assertEqual(code, 0)
        self.assertIn("machaon demo", out)
        self.assertIn("machaon teach", out)
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
