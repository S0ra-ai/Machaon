"""The offline real-model example exercises the whole program lifecycle."""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "tiny_program.py"

try:
    import tokenizers  # noqa: F401
    import torch  # noqa: F401
    import transformers  # noqa: F401
    HAVE_RUNTIME = True
except ImportError:
    HAVE_RUNTIME = False

MPS_AVAILABLE = bool(
    HAVE_RUNTIME
    and hasattr(torch.backends, "mps")
    and torch.backends.mps.is_available()
)


def _example_module():
    spec = importlib.util.spec_from_file_location(
        "machaon_tiny_program_example", EXAMPLE
    )
    if spec is None or spec.loader is None:
        raise ImportError("cannot load %s" % EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(HAVE_RUNTIME, "torch + transformers + tokenizers not installed")
class TestTinyProgramExample(unittest.TestCase):
    def test_program_saves_reloads_and_report_matches(self):
        example = _example_module()
        with tempfile.TemporaryDirectory() as tmp:
            payload = example.run_demo(Path(tmp))
        self.assertEqual(payload["status"], "applied")
        self.assertTrue(payload["accepted"])
        self.assertTrue(payload["verification"]["passed"])
        self.assertEqual(len(payload["report_sha256"]), 64)
        checked = payload["checked"]["model_check"]
        self.assertTrue(checked["performed"])
        self.assertTrue(checked["passed"])
        self.assertTrue(checked["matches_report"])

    @unittest.skipUnless(MPS_AVAILABLE, "MPS is not available")
    def test_program_lifecycle_on_mps(self):
        example = _example_module()
        with tempfile.TemporaryDirectory() as tmp:
            payload = example.run_demo(Path(tmp), device="mps")
        self.assertEqual(payload["status"], "applied")
        self.assertEqual(payload["device"], "mps")
        self.assertTrue(payload["checked"]["model_check"]["passed"])


if __name__ == "__main__":
    unittest.main()
