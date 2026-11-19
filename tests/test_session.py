"""The agent-facing loop: baseline -> propose -> apply -> evaluate -> rollback."""
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import numpy as np

try:
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    from machaon.budget import BudgetExceeded, EditBudget
    from machaon.headroom import DecodingConfig
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

    def decode(self, tokens, skip_special_tokens=True):
        return "tokens:" + ",".join(str(int(value)) for value in tokens.tolist())


class FlakyRuntime:
    """Delegate to HFModel, but fail writes when asked."""

    def __init__(self, inner, fail_writes=False):
        self.inner = inner
        self.fail_writes = fail_writes

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def write(self, name, values):
        if self.fail_writes:
            raise OSError("disk full")
        return self.inner.write(name, values)


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
        self.assertEqual(state.device, "cpu")
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
        changes = self.session.changes()
        self.assertIn("model.layers.1.mlp.down_proj.weight", changes)
        self.assertEqual(
            changes["model.layers.1.mlp.down_proj.weight"].shape,
            self.runtime.read("model.layers.1.mlp.down_proj.weight").shape,
        )
        self.assertEqual(self.session.state().edited,
                         ["model.layers.1.mlp.down_proj.weight"])

    def test_wsp_evidence_restores_the_candidate_state(self):
        self.session.baseline(["a", "b"])
        self.session.propose(
            "anything", "", layer=1, target_token_id=self.token,
            solver="direction", alphas=(0.5, 1, 2, 4, 8, 16, 32, 64))
        self.session.apply()
        before = self.session.changes()

        evidence = self.session.evaluate_wsp(["anything"], [self.token])
        self.assertEqual(set(evidence), {"base", "candidate", "controls"})
        self.assertIn("random_same_support", evidence["controls"])
        self.assertEqual(evidence["candidate"]["accuracy"], 1.0)

        after = self.session.changes()
        self.assertEqual(set(after), set(before))
        for name in before:
            np.testing.assert_array_equal(after[name], before[name])

        report = self.session.assess_wsp(
            ["anything"], [self.token], headroom={"accuracy": 0.99}
        )
        self.assertEqual(report["label"], "repair")
        self.assertFalse(report["protocol"]["require_gates"])

        calls = []
        report = self.session.assess_wsp(
            ["anything"], [self.token],
            headroom_runner=lambda runtime: calls.append(runtime) or {"accuracy": 0.99},
        )
        self.assertEqual(report["label"], "repair")
        self.assertEqual(report["classification"]["headroom_name"], "headroom")
        self.assertEqual(len(calls), 1)
        for name in before:
            np.testing.assert_array_equal(
                self.session.changes()[name], before[name]
            )

    def test_propose_batch_applies_one_delta(self):
        session = Session(self.runtime, "toy", EditBudget(max_total=100.0))
        name = "model.layers.1.mlp.down_proj.weight"
        original = self.runtime.read(name)
        proposal = session.propose_batch(
            ["a", "b"], np.zeros((2, 64)), layer=1
        )
        self.assertTrue(proposal["exact"])
        self.assertFalse(proposal["clamped"])
        self.assertEqual(proposal["module"], name)
        np.testing.assert_array_equal(self.runtime.read(name), original)
        self.assertTrue(session.state().has_pending)

        applied = session.apply()
        self.assertTrue(applied["exact"])
        self.assertEqual(applied["module"], name)
        self.assertIn(name, session.changes())
        self.assertFalse(session.state().has_pending)

        session.rollback()
        np.testing.assert_array_equal(self.runtime.read(name), original)

    def test_propose_action_applies_and_rolls_back(self):
        session = Session(self.runtime, "toy", EditBudget(max_total=1.0))
        name = "model.layers.1.mlp.down_proj.weight"
        original = self.runtime.read(name)
        proposal = session.propose_action(
            "def edit(W):\n    return 0.001 * W\n",
            "scale tensor for action test",
            layer=1,
            max_rel_norm=0.01,
        )
        self.assertEqual(proposal["module"], name)
        self.assertAlmostEqual(proposal["relative_delta"], 0.001, places=6)
        self.assertEqual(len(proposal["source_sha256"]), 64)
        self.assertTrue(session.state().has_pending)
        np.testing.assert_array_equal(self.runtime.read(name), original)

        applied = session.apply()
        self.assertEqual(applied["source_sha256"], proposal["source_sha256"])
        np.testing.assert_allclose(
            self.runtime.read(name), original * 1.001, rtol=1e-6, atol=1e-7
        )
        self.assertFalse(session.state().has_pending)

        session.rollback()
        np.testing.assert_array_equal(self.runtime.read(name), original)

    def test_assess_generation_wsp_restores_candidate(self):
        session = Session(self.runtime, "toy", EditBudget(max_total=1.0))
        session.propose_action(
            "def edit(W):\n    return 0.001 * W\n",
            "generation WSP test",
            layer=1,
            max_rel_norm=0.01,
        )
        session.apply()
        before = session.changes()
        report = session.assess_generation_wsp(
            ["anything"],
            [""],
            config=DecodingConfig(max_new_tokens=2),
            headroom_config=DecodingConfig(max_new_tokens=3),
        )
        self.assertIn("mechanism", report)
        self.assertIn("metrics", report["mechanism"])
        self.assertIn("loop_rate", report["mechanism"]["metrics"])
        after = session.changes()
        self.assertEqual(set(after), set(before))
        for name in before:
            np.testing.assert_array_equal(after[name], before[name])

    def test_search_layers_rolls_back_every_trial(self):
        session = Session(self.runtime, "toy", EditBudget(max_total=1.0))
        name = "model.layers.0.mlp.down_proj.weight"
        original = self.runtime.read(name)

        def propose(session, layer):
            session.propose_batch(
                ["anything"], np.zeros((1, 64)), layer=layer
            )

        report = session.search_layers(
            [0, 1],
            propose,
            lambda s: s.evaluate(["anything"], [self.token]),
            metric="accuracy",
        )
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["trials"]), 2)
        self.assertEqual(session.state().edited, [])
        self.assertEqual(session.state().budget_total, 0.0)
        np.testing.assert_array_equal(self.runtime.read(name), original)

    def test_search_rome_layers_builds_a_rome_proposal_per_layer(self):
        session = Session(self.runtime, "toy", EditBudget(max_total=1.0))

        def fake_search(layers, proposal_factory, evaluation, **kwargs):
            proposal_factory(session, 7)
            evaluation(session)
            return {"passed": True, "layers": list(layers), "kwargs": kwargs}

        with mock.patch.object(session, "search_layers", side_effect=fake_search), \
                mock.patch.object(session, "propose") as propose, \
                mock.patch.object(session, "evaluate",
                                  return_value={"accuracy": 1.0}) as evaluate:
            result = session.search_rome_layers(
                "prompt", "target", [7], max_rel_norm=0.01, metric="accuracy"
            )
        propose.assert_called_once()
        self.assertEqual(propose.call_args.args[:3], ("prompt", "target", 7))
        self.assertEqual(propose.call_args.kwargs["max_rel_norm"], 0.01)
        evaluate.assert_called_once_with(["prompt"], ["target"])
        self.assertEqual(result["search"], "rome")
        self.assertEqual(result["layers"], [7])

    def test_export_openwsp_binds_metadata(self):
        self.session.propose(
            "anything", "", layer=1, target_token_id=self.token,
            solver="direction", alphas=(0.5, 1, 2, 4, 8, 16, 32, 64))
        self.session.apply()
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self.session.export_openwsp(
                tmp,
                base_model="toy",
                metadata={"evidence_sha256": "evidence123"},
            )
            self.assertEqual(manifest["format"], "openwsp.edit")
            self.assertEqual(manifest["base_model"], "toy")
            self.assertEqual(
                manifest["machaon"]["evidence_sha256"], "evidence123"
            )
            self.assertTrue((Path(tmp) / "manifest.json").exists())

    def test_export_openwsp_lowrank_form(self):
        self.session.propose(
            "anything", "", layer=1, target_token_id=self.token,
            solver="direction", alphas=(0.5, 1, 2, 4, 8, 16, 32, 64))
        self.session.apply()
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self.session.export_openwsp(
                tmp, base_model="toy", form="lowrank", rank=1
            )
            self.assertEqual(manifest["tensors"][0]["form"], "lowrank")
            self.assertEqual(manifest["tensors"][0]["rank"], 1)

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

    def test_failed_write_does_not_charge_budget(self):
        flaky = FlakyRuntime(self.runtime, fail_writes=True)
        session = Session(flaky, "toy", EditBudget(max_total=1.0))
        session.propose(
            "anything", "", layer=1, target_token_id=self.token,
            solver="direction", alphas=(0.5, 1, 2, 4, 8, 16, 32, 64))
        with self.assertRaises(OSError):
            session.apply()
        self.assertEqual(session.state().budget_total, 0.0)

    def test_failed_rollback_keeps_the_original_for_a_retry(self):
        self.session.propose(
            "anything", "", layer=1, target_token_id=self.token,
            solver="direction", alphas=(0.5, 1, 2, 4, 8, 16, 32, 64))
        self.session.apply()
        flaky = FlakyRuntime(self.runtime, fail_writes=True)
        self.session.runtime = flaky
        with self.assertRaises(OSError):
            self.session.rollback()
        self.assertEqual(self.session.state().edited,
                         ["model.layers.1.mlp.down_proj.weight"])


class TestQuantizedOpen(unittest.TestCase):
    """Quantized loading passes the documented Quanto configuration through."""

    def test_quantize_uses_quanto_and_skips_a_second_move(self):
        import machaon.runtime.hf as hf
        import machaon.session as session_module

        torch_module = ModuleType("torch")
        torch_module.float32 = "float32"
        torch_module.float16 = "float16"
        torch_module.bfloat16 = "bfloat16"

        transformers_module = ModuleType("transformers")
        transformers_module.AutoTokenizer = SimpleNamespace(
            from_pretrained=lambda _path: object()
        )

        class QuantoConfig:
            def __init__(self, weights):
                self.weights = weights

        transformers_module.QuantoConfig = QuantoConfig
        captured = {}

        class FakeModel:
            def eval(self):
                return self

        def loader(_path, **kwargs):
            captured.update(kwargs)
            return FakeModel()

        class FakeRuntime:
            def __init__(self, model, tokenizer):
                self.model = model
                self.tokenizer = tokenizer

        with mock.patch.dict(
            "sys.modules",
            {"torch": torch_module, "transformers": transformers_module},
        ), mock.patch.object(hf, "load_pretrained", loader), mock.patch.object(
            session_module, "HFModel", FakeRuntime
        ):
            session = session_module.Session.open(
                "/tmp/model", dtype="auto", quantize="int4"
            )
            with self.assertRaises(ValueError):
                session_module.Session.open(
                    "/tmp/model", dtype="float32", quantize="int4"
                )

        self.assertEqual(session.quantize, "int4")
        self.assertEqual(captured["device_map"], "cpu")
        self.assertEqual(captured["quantization_config"].weights, "int4")


class TestBrokenCheckpoint(unittest.TestCase):
    def test_interrupted_download_fails_before_the_tokenizer(self):
        import machaon.session as session_module

        torch_module = ModuleType("torch")
        torch_module.float32 = "float32"
        torch_module.float16 = "float16"
        torch_module.bfloat16 = "bfloat16"

        loaded = []
        transformers_module = ModuleType("transformers")
        transformers_module.AutoTokenizer = SimpleNamespace(
            from_pretrained=lambda path: loaded.append(path)
        )

        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model"
            model.mkdir()
            (model / "config.json").write_text("{}")
            (model / "model.safetensors.part").write_bytes(b"")
            with mock.patch.dict(
                "sys.modules",
                {"torch": torch_module, "transformers": transformers_module},
            ):
                with self.assertRaisesRegex(ValueError, "unfinished download"):
                    session_module.Session.open(str(model))
        self.assertEqual(loaded, [])


if __name__ == "__main__":
    unittest.main()
