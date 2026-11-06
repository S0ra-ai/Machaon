"""Causal tracing on a tiny model: no downloads, no GPU."""
import unittest

import numpy as np

try:
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    from machaon.podalirius import find_layers, patch_scores
    HAVE_HF = True
except ImportError:
    HAVE_HF = False

requires_hf = unittest.skipUnless(HAVE_HF, "torch + transformers not installed")


def build():
    torch.manual_seed(0)
    return GPT2LMHeadModel(GPT2Config(
        vocab_size=96, n_positions=32, n_embd=32, n_layer=3, n_head=4,
        tie_word_embeddings=False)).eval()


@requires_hf
class TestPatchScores(unittest.TestCase):
    def setUp(self):
        self.model = build()
        self.clean = torch.tensor([[5, 9, 14, 22]])
        self.corrupt = torch.tensor([[5, 9, 14, 7]])

    def test_scores_every_layer_and_ranks_them(self):
        report = patch_scores(self.model, self.clean, self.corrupt, target_token_id=22)
        self.assertEqual(report["layer_path"], "transformer.h")
        self.assertEqual(report["n_layers"], 3)
        self.assertEqual(sorted(report["scores"]), [0, 1, 2])
        self.assertEqual(len(report["ranked"]), 3)
        self.assertIn(report["best_layer"], report["scores"])
        for row in report["scores"].values():
            self.assertTrue(np.isfinite(row["p_target"]))
            self.assertTrue(np.isfinite(row["recovery"]))

    def test_selected_layers_only(self):
        report = patch_scores(
            self.model, self.clean, self.corrupt, target_token_id=22, layers=[1])
        self.assertEqual(sorted(report["scores"]), [1])
        self.assertEqual(report["best_layer"], 1)

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            patch_scores(self.model, self.clean, torch.tensor([[5, 9]]), 22)

    def test_out_of_range_layer_is_rejected(self):
        with self.assertRaises(IndexError):
            patch_scores(self.model, self.clean, self.corrupt, 22, layers=[9])

    def test_find_layers_finds_a_known_family(self):
        layers, path = find_layers(self.model)
        self.assertEqual(path, "transformer.h")
        self.assertEqual(len(layers), 3)


if __name__ == "__main__":
    unittest.main()
