"""Explain broken checkpoint directories before a tokenizer traceback."""
import tempfile
import unittest
from pathlib import Path

from machaon.checkpointing import checkpoint_problem


class TestCheckpointProblem(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def _complete(self) -> Path:
        model = self.root / "model"
        model.mkdir()
        (model / "config.json").write_text("{}")
        (model / "model.safetensors").write_bytes(b"")
        (model / "tokenizer.json").write_text("{}")
        return model

    def test_a_complete_checkpoint_has_no_problem(self):
        self.assertEqual(checkpoint_problem(self._complete()), "")

    def test_an_interrupted_download_is_named(self):
        model = self._complete()
        (model / "model.safetensors.part").write_bytes(b"")
        self.assertIn("unfinished download", checkpoint_problem(model))

    def test_missing_config_weights_and_tokenizer_are_distinguished(self):
        model = self.root / "bare"
        model.mkdir()
        self.assertIn("config.json", checkpoint_problem(model))
        (model / "config.json").write_text("{}")
        self.assertIn("no weight files", checkpoint_problem(model))
        (model / "model.safetensors").write_bytes(b"")
        self.assertIn("no tokenizer files", checkpoint_problem(model))

    def test_a_missing_directory_is_left_to_the_caller(self):
        self.assertEqual(checkpoint_problem(self.root / "absent"), "")


if __name__ == "__main__":
    unittest.main()
