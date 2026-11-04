"""The agent-facing loop: baseline -> propose -> apply -> evaluate -> rollback."""
import unittest

import numpy as np

try:
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    from machaon.budget import BudgetExceeded, EditBudget
    from machaon.runtime import HFModel
    from machaon.session import Session
    HAVE_HF = True
except ImportError:
    HAVE_HF = False

requires_hf = unittest.skipUnless(HAVE_HF, "torch + transformers not installed")


class StubTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [5, 6, 7]}


def build(seed=0):
    torch.manual_seed(seed)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=96, hidden_size=64, intermediate_size=128,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=64)).eval()
    runtime = HFModel(model)
    runtime.tokenizer = StubTokenizer()
    return runtime


@requires_hf
class TestSession(unittest.TestCase):
    def setUp(self):
        self.runtime = build()
        self.session = Session(self.runtime, "toy", EditBudget(max_total=1.0))
        self.token = int(np.argsort(self.runtime.next_logits(["anything"])[0])[-4])

    def test_baseline_and_state(self):
        result = self.session.baseline(["a", "b"], ["c d e"])
        self.assertEqual(result["prompts"], 2)
        self.assertEqual(result["corpus"], 1)
        state = self.session.state()
        self.assertTrue(state.has_baseline)
        self.assertEqual(state.edited, [])

    def test_propose_does_not_write(self):
        before = self.runtime.read("model.layers.1.mlp.down_proj.weight")
        proposal = self.session.propose(
            "anything", "", layer=1, target_token_id=self.token,
            solver="direction", alphas=(0.5, 1, 2, 4, 8, 16, 32, 64))
        after = self.runtime.read("model.layers.1.mlp.down_proj.weight")
        np.testing.assert_array_equal(before, after)
        self.assertGreater(proposal["relative_delta"], 0.0)
        self.assertEqual(proposal["module"], "model.layers.1.mlp.down_proj.weight")

    def test_apply_changes_prediction_and_evaluate_sees_it(self):
        self.session.baseline(["a", "b"])
        self.session.propose(
            "anything", "", layer=1, target_token_id=self.token,
            solver="direction", alphas=(0.5, 1, 2, 4, 8, 16, 32, 64))
        self.session.apply()
        metrics = self.session.evaluate(["anything"], [self.token])
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertIn("kl_p95", metrics)
        self.assertEqual(self.session.state().edited,
                         ["model.layers.1.mlp.down_proj.weight"])

    def test_rollback_restores_bytes(self):
        original = self.runtime.read("model.layers.1.mlp.down_proj.weight")
        self.session.propose(
            "anything", "", layer=1, target_token_id=self.token,
            solver="direction", alphas=(0.5, 1, 2, 4, 8, 16, 32, 64))
        self.session.apply()
        self.session.rollback()
        np.testing.assert_array_equal(
            self.runtime.read("model.layers.1.mlp.down_proj.weight"), original)
        self.assertEqual(self.session.state().edited, [])
        self.assertEqual(self.session.state().budget_total, 0.0)

    def test_budget_refuses_apply(self):
        session = Session(self.runtime, "toy", EditBudget(max_total=1e-9))
        original = self.runtime.read("model.layers.1.mlp.down_proj.weight")
        session.propose(
            "anything", "", layer=1, target_token_id=self.token,
            solver="direction", alphas=(0.5, 1, 2, 4, 8, 16, 32, 64))
        with self.assertRaises(BudgetExceeded):
            session.apply()
        np.testing.assert_array_equal(
            self.runtime.read("model.layers.1.mlp.down_proj.weight"), original)

    def test_apply_without_proposal_is_rejected(self):
        with self.assertRaises(RuntimeError):
            self.session.apply()


if __name__ == "__main__":
    unittest.main()
