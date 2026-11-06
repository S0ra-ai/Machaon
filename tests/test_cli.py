"""The CLI is a public interface: exit codes and JSON output are tested."""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest

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

    def test_doctor_json_has_core_and_optional_sections(self):
        code, out, _ = run("doctor", "--json")
        self.assertEqual(code, 0)
        report = json.loads(out)
        self.assertEqual(report["machaon"], __import__("machaon").__version__)
        self.assertTrue(all(report["core"].values()))
        self.assertIn("torch", report["packages"])

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
