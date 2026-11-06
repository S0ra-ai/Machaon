"""Runtime + evaluation integration on a tiny random model. No network."""
import copy
import unittest

import numpy as np

try:
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM

    from machaon.hygieia import EditSuite, evaluate_edit
    from machaon.hygieia.gates import evaluate
    from machaon.runtime import HFModel
    HAVE_HF = True
except ImportError:
    HAVE_HF = False

requires_hf = unittest.skipUnless(HAVE_HF, "torch + transformers not installed")

GATES = {"rewrite_acc": (">=", 0.99), "kl_p95": ("<=", 0.05), "ppl_ratio": ("<=", 1.05)}


def build(seed=0):
    torch.manual_seed(seed)
    return GPT2LMHeadModel(GPT2Config(
        vocab_size=96, n_positions=32, n_embd=48, n_layer=3, n_head=4,
        tie_word_embeddings=False)).eval()


@requires_hf
class TestRuntime(unittest.TestCase):
    def setUp(self):
        self.model = build()
        self.rt = HFModel(self.model)

    def test_layer_discovery(self):
        self.assertEqual(self.rt.n_layers, 3)
        self.assertEqual(self.rt.layer_path, "transformer.h")

    def test_edit_target_translates_down_proj_to_c_proj(self):
        self.assertEqual(self.rt.edit_target(1), "transformer.h.1.mlp.c_proj.weight")
        self.assertEqual(self.rt.edit_target(1, "mlp.c_proj"), "transformer.h.1.mlp.c_proj.weight")

    def test_conv1d_orientation_is_normalised(self):
        name = "transformer.h.1.mlp.c_proj.weight"
        stored = tuple(self.model.transformer.h[1].mlp.c_proj.weight.shape)
        self.assertEqual(self.rt.weight_shape(name), tuple(reversed(stored)))
        self.assertEqual(self.rt.read(name).shape, tuple(reversed(stored)))

    def test_read_write_roundtrip(self):
        name = "transformer.h.1.mlp.c_proj.weight"
        original = self.rt.read(name)
        self.assertEqual(original.dtype, np.float64)
        self.rt.write(name, original + 1e-3)
        np.testing.assert_allclose(self.rt.read(name), original + 1e-3, atol=1e-6)
        self.rt.write(name, original)
        np.testing.assert_allclose(self.rt.read(name), original, atol=0)

    def test_densify_is_a_noop_on_dense_layers(self):
        self.assertFalse(self.rt.densify("transformer.h.1.mlp.c_proj.weight"))

    def test_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            self.rt.write("transformer.h.1.mlp.c_proj.weight", np.zeros((2, 2)))

    def test_logits_and_nll(self):
        ids = np.array([[5, 9, 14, 22]])
        logits = self.rt.next_logits([ids])
        self.assertEqual(logits.shape, (1, 96))
        nll = self.rt.nll([ids])
        self.assertEqual(nll.shape, (1,))
        self.assertGreater(nll[0], 0.0)

    def test_ragged_token_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            self.rt.next_logits([np.array([1, 2, 3]), np.array([4, 5])])

    def test_nll_needs_a_real_prefix(self):
        with self.assertRaises(ValueError):
            self.rt.nll([np.array([[7]])])

    def test_capture_module_input(self):
        ids = np.array([[5, 9, 14, 22]])
        x = self.rt.capture_module_input("lm_head", ids)
        self.assertEqual(x.shape, (1, 48))
        self.assertTrue(np.isfinite(x).all())


class _StubTokenizer:
    """One id per whitespace token, no pad token, like GPT-2."""

    pad_token_id = None
    eos_token_id = 0

    def __call__(self, text):
        return {"input_ids": list(range(1, len(text.split()) + 1))}


def build_llama(seed=0):
    torch.manual_seed(seed)
    return LlamaForCausalLM(LlamaConfig(
        vocab_size=96, hidden_size=64, intermediate_size=128,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=64)).eval()


@requires_hf
class TestLlamaStyle(unittest.TestCase):
    """A second architecture family: `model.layers` + `down_proj` + nn.Linear."""

    def setUp(self):
        self.model = build_llama()
        self.rt = HFModel(self.model)

    def test_layer_discovery(self):
        self.assertEqual(self.rt.layer_path, "model.layers")
        self.assertEqual(self.rt.n_layers, 3)

    def test_edit_target_uses_down_proj(self):
        self.assertEqual(self.rt.edit_target(1), "model.layers.1.mlp.down_proj.weight")

    def test_linear_orientation_is_preserved(self):
        name = "model.layers.1.mlp.down_proj.weight"
        stored = tuple(self.model.model.layers[1].mlp.down_proj.weight.shape)
        self.assertEqual(self.rt.weight_shape(name), stored)
        self.assertEqual(stored, (64, 128))

    def test_read_write_roundtrip(self):
        name = self.rt.edit_target(2)
        original = self.rt.read(name)
        self.rt.write(name, original + 1e-3)
        np.testing.assert_allclose(self.rt.read(name), original + 1e-3, atol=1e-6)
        self.rt.write(name, original)
        np.testing.assert_allclose(self.rt.read(name), original, atol=0)

    def test_forward_contract(self):
        ids = np.array([[5, 9, 14, 22]])
        self.assertEqual(self.rt.next_logits([ids]).shape, (1, 96))
        self.assertEqual(self.rt.nll([ids]).shape, (1,))

    def test_same_edit_pipeline_works(self):
        before = HFModel(build_llama())
        edited = HFModel(build_llama())
        target = np.array([[5, 9, 14, 22]])
        token = int(np.argsort(before.next_logits([target])[0])[-2])
        hidden = before.capture_module_input("lm_head", target)[0]
        base = before.read("lm_head.weight")
        direction = np.zeros_like(base[:, 0])
        direction[token] = 12.0
        edited.write("lm_head.weight", base + np.outer(direction, hidden) / float(hidden @ hidden))

        metrics = evaluate_edit(
            before, edited,
            EditSuite([target], [token], corpus_items=[np.array([[1, 4, 6, 9, 12]])]),
        )
        self.assertEqual(metrics["rewrite_acc"], 1.0)
        self.assertIn("ppl_ratio", metrics)


@requires_hf
class TestRomeProcedure(unittest.TestCase):
    """The procedure, end to end, on a tiny Llama. No downloads."""

    def setUp(self):
        self.rt = HFModel(build_llama())

        class Stub:
            pad_token_id = 0
            eos_token_id = 0

            def __call__(self, text, add_special_tokens=True):
                return {"input_ids": [5, 6, 7]}

        self.rt.tokenizer = Stub()

    def test_direction_solver_flips_the_prediction(self):
        from machaon.procedures import rome_edit

        before = self.rt.next_logits([[5, 6, 7]])[0]
        token = int(np.argsort(before)[-5])
        result = rome_edit(
            self.rt, "anything", "", layer=1, target_token_id=token,
            solver="direction", alphas=(0.5, 1, 2, 4, 8, 16, 32, 64),
        )
        after = self.rt.next_logits([[5, 6, 7]])[0]
        self.assertEqual(int(after.argmax()), token)
        self.assertGreater(result.relative_delta, 0.0)
        self.assertEqual(result.solver, "direction")

    def test_gradient_solver_runs(self):
        from machaon.procedures import rome_edit

        token = int(np.argsort(self.rt.next_logits([[5, 6, 7]])[0])[-3])
        result = rome_edit(
            self.rt, "anything", "", layer=1, target_token_id=token,
            solver="gradient", steps=5,
        )
        self.assertEqual(result.steps, 5)
        self.assertGreater(result.relative_delta, 0.0)

    def test_unknown_solver_rejected(self):
        from machaon.procedures import rome_edit

        with self.assertRaises(ValueError):
            rome_edit(self.rt, "x", "", layer=1, target_token_id=1, solver="magic")


@requires_hf
class TestPadding(unittest.TestCase):
    def test_manual_padding_produces_a_mask(self):
        rt = HFModel(build())
        rt.tokenizer = _StubTokenizer()
        ids, mask = rt.encode(["a b c", "a b"])
        self.assertEqual(tuple(ids.shape), (2, 3))
        self.assertEqual(mask.tolist(), [[1, 1, 1], [1, 1, 0]])

    def test_text_and_ids_agree_on_the_last_real_token(self):
        rt = HFModel(build())
        rt.tokenizer = _StubTokenizer()
        ids, mask = rt.encode(["a b c", "a b"])
        batched = rt.next_logits(["a b c", "a b"])
        alone = rt.next_logits_ids([[1, 2, 3]])
        np.testing.assert_allclose(batched[0], alone[0], rtol=1e-5, atol=1e-5)

    def test_nll_ignores_padding(self):
        rt = HFModel(build())
        rt.tokenizer = _StubTokenizer()
        batched = rt.nll(["a b c", "a b"])
        alone = rt.nll_ids([[1, 2]])
        np.testing.assert_allclose(batched[1], alone[0], rtol=1e-5, atol=1e-5)


@requires_hf
class TestEndToEnd(unittest.TestCase):
    def test_budget_decides_the_verdict(self):
        model = build()
        before = HFModel(model)
        target = torch.tensor([[5, 9, 14, 22]])
        neutral = torch.tensor([[3, 7, 11], [2, 8, 13], [17, 19, 23]])
        corpus = torch.tensor([[1, 4, 6, 9, 12], [8, 10, 15, 18, 21]])

        token = int(np.argsort(before.next_logits_ids(target)[0])[-3])
        hidden = before.capture_module_input("lm_head", target)[0]
        base = before.read("lm_head.weight")

        def run(beta):
            after = HFModel(copy.deepcopy(model))
            direction = np.zeros_like(base[:, 0])
            direction[token] = beta
            after.write("lm_head.weight", base + np.outer(direction, hidden) / float(hidden @ hidden))
            metrics = evaluate_edit(
                before, after, EditSuite([target], [token], [neutral], [corpus]))
            return metrics, evaluate(metrics, GATES)

        cautious, verdict_small = run(1.0)
        greedy, verdict_large = run(12.0)

        self.assertEqual(cautious["rewrite_acc"], 1.0)
        self.assertEqual(greedy["rewrite_acc"], 1.0)
        self.assertTrue(verdict_small.passed, verdict_small.summary())
        self.assertFalse(verdict_large.passed)
        self.assertGreater(greedy["ppl_ratio"], cautious["ppl_ratio"])


if __name__ == "__main__":
    unittest.main()
