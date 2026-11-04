"""The same edit path across model families, at tiny scale.

No downloads: each family is instantiated from a config. What is being checked
is the part that differs between architectures — where the decoder blocks live,
what the MLP output projection is called, and how the weight is stored.
"""
import unittest

import numpy as np

try:
    import torch
    from transformers import (
        GemmaConfig,
        GemmaForCausalLM,
        LlamaConfig,
        LlamaForCausalLM,
        MistralConfig,
        MistralForCausalLM,
        Phi3Config,
        Phi3ForCausalLM,
        Qwen3Config,
        Qwen3ForCausalLM,
    )

    from machaon.runtime import HFModel
    HAVE_HF = True
except ImportError:
    HAVE_HF = False

requires_hf = unittest.skipUnless(HAVE_HF, "torch + transformers not installed")

COMMON = dict(vocab_size=96, hidden_size=64, intermediate_size=128,
              num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4,
              max_position_embeddings=64)

FAMILIES = {}
if HAVE_HF:
    FAMILIES = {
        "llama": (LlamaConfig, LlamaForCausalLM),
        "mistral": (MistralConfig, MistralForCausalLM),
        "gemma": (GemmaConfig, GemmaForCausalLM),
        "qwen3": (Qwen3Config, Qwen3ForCausalLM),
        "phi3": (Phi3Config, Phi3ForCausalLM),
    }


def build(name):
    config_cls, model_cls = FAMILIES[name]
    torch.manual_seed(0)
    kwargs = dict(COMMON)
    if name == "phi3":
        kwargs["pad_token_id"] = 0
    return model_cls(config_cls(**kwargs)).eval()


@requires_hf
class TestFamilies(unittest.TestCase):
    def test_layer_layout_and_target_name(self):
        for name in FAMILIES:
            with self.subTest(family=name):
                runtime = HFModel(build(name))
                self.assertEqual(runtime.layer_path, "model.layers")
                self.assertEqual(runtime.n_layers, 3)
                self.assertEqual(runtime.edit_target(1),
                                 "model.layers.1.mlp.down_proj.weight")
                self.assertEqual(runtime.weight_shape(runtime.edit_target(1)), (64, 128))

    def test_read_write_roundtrip(self):
        for name in FAMILIES:
            with self.subTest(family=name):
                runtime = HFModel(build(name))
                target = runtime.edit_target(2)
                original = runtime.read(target)
                runtime.write(target, original + 1e-3)
                np.testing.assert_allclose(runtime.read(target), original + 1e-3, atol=1e-6)
                runtime.write(target, original)
                np.testing.assert_allclose(runtime.read(target), original, atol=0)

    def test_densify_is_a_noop(self):
        for name in FAMILIES:
            with self.subTest(family=name):
                runtime = HFModel(build(name))
                self.assertFalse(runtime.densify(runtime.edit_target(1)))

    def test_forward_contract(self):
        for name in FAMILIES:
            with self.subTest(family=name):
                runtime = HFModel(build(name))
                ids = np.array([[5, 9, 14, 22]])
                self.assertEqual(runtime.next_logits([ids]).shape, (1, 96))
                self.assertEqual(runtime.nll([ids]).shape, (1,))
                captured = runtime.capture_module_input("model.layers.1.mlp.down_proj", ids)
                self.assertEqual(captured.shape, (1, 128))


if __name__ == "__main__":
    unittest.main()
