"""The one-call teaching recipe, without loading a model."""
import unittest

import numpy as np

from machaon.recipes import (
    DEFAULT_BASELINE_PROMPTS,
    TeachPolicy,
    candidate_layers,
    teach,
    teach_fact,
)

try:
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    from machaon.budget import EditBudget
    from machaon.runtime import HFModel
    from machaon.session import Session

    HAVE_HF = True
except ImportError:
    HAVE_HF = False

requires_hf = unittest.skipUnless(HAVE_HF, "torch + transformers not installed")


class FakeTokenizer:
    def __init__(self):
        self.vocab = {}
        self.reverse = {}

    def _id(self, text):
        if text not in self.vocab:
            token = len(self.vocab) + 2
            self.vocab[text] = token
            self.reverse[token] = text
        return self.vocab[text]

    def __call__(self, text, add_special_tokens=False):
        ids = [self._id(text)]
        if text == "New York":
            ids.append(self._id(" York"))
        return {"input_ids": ids}

    def convert_ids_to_tokens(self, token):
        return self.reverse.get(int(token), "tok%d" % token)


class FakeRuntime:
    def __init__(self, tokenizer, n_layers=8):
        self.tokenizer = tokenizer
        self.n_layers = n_layers
        self.learned = False
        self.panel_shift = False
        self.target_token = None
        self.target_prompt = None
        self.vocab_size = 64
        self.tensors = {
            "model.layers.%d.mlp.down_proj.weight" % layer: np.zeros((2, 2))
            for layer in range(n_layers)
        }

    def next_logits(self, prompts):
        rows = []
        for prompt in prompts:
            logits = np.full(self.vocab_size, -8.0)
            if prompt == self.target_prompt and self.learned:
                logits[self.target_token] = 8.0
            elif prompt in DEFAULT_BASELINE_PROMPTS and self.panel_shift:
                # A changed argmax on every panel prompt after the edit.
                logits[3 if not self.learned else 4] = 8.0
            else:
                logits[2] = 8.0
            rows.append(logits)
        return np.stack(rows)

    def read(self, name):
        return self.tensors[name].copy()

    def write(self, name, value):
        self.tensors[name] = np.asarray(value, dtype=float).copy()


class FakeState:
    def __init__(self, session):
        self.session = session
        self.edited = list(session.edited)
        self.has_pending = session.pending is not None
        self.has_baseline = session.baseline_captured


class FakeSession:
    def __init__(self, *, winning=(5,), base_correct=False, kl=0.001,
                 panel_shift=False, fail_apply=False):
        self.runtime = FakeRuntime(FakeTokenizer())
        self.model_id = "fake-model"
        self.winning = set(winning)
        self.base_correct = base_correct
        self.kl = kl
        self.runtime.panel_shift = panel_shift
        self.fail_apply = fail_apply
        self.edited = []
        self.pending = None
        self.baseline_captured = False
        self.budget_total = 0.0
        self.search_calls = 0
        self.prompt = None
        self.answer_token = None

    def state(self):
        return FakeState(self)

    def baseline(self, prompts, corpus=()):
        self.baseline_captured = True
        return {"prompts": len(prompts), "corpus": len(corpus)}

    def evaluate(self, prompts, targets):
        prompt = prompts[0]
        token = int(targets[0])
        accuracy = 1.0 if (
            (prompt == self.prompt and token == self.answer_token
             and (self.edited or self.runtime.learned))
            or (self.base_correct and not self.edited)
        ) else 0.0
        metrics = {"accuracy": accuracy, "n": 1}
        if self.baseline_captured:
            metrics["kl_p95"] = 0.0 if self.base_correct else float(self.kl)
        return metrics

    def search_rome_layers(self, prompt, target, layers, **kwargs):
        self.search_calls += 1
        self.prompt = prompt
        self.answer_token = self.runtime.tokenizer(target)["input_ids"][0]
        trials = []
        for layer in layers:
            ok = int(layer) in self.winning
            trials.append({
                "layer": int(layer),
                "status": "ok",
                "score": 1.0 if ok else 0.0,
                "metrics": {
                    "accuracy": 1.0 if ok else 0.0,
                    "kl_p95": 0.001 + 0.001 * int(layer) if ok else 0.5,
                    "relative_delta": 0.01 if ok else 0.02,
                },
                "error": None,
            })
        ranking = [
            {"layer": int(t["layer"]), "score": t["score"]}
            for t in trials if t["status"] == "ok"
        ]
        return {
            "trials": trials,
            "ranking": ranking,
            "best_layer": int(self.winning and sorted(self.winning)[0])
            if self.winning else None,
        }

    def propose(self, prompt, target, layer, **kwargs):
        self.prompt = prompt
        self.answer_token = self.runtime.tokenizer(target)["input_ids"][0]
        self.pending = {
            "module": "model.layers.%d.mlp.down_proj.weight" % int(layer),
            "relative_delta": 0.01,
            "target_probability": 0.99,
        }
        return dict(self.pending)

    def apply(self):
        if self.fail_apply:
            raise OSError("disk full")
        if self.pending is None:
            raise RuntimeError("no pending proposal")
        name = self.pending["module"]
        self.edited = [name]
        self.budget_total = float(self.pending["relative_delta"])
        self.runtime.learned = True
        self.runtime.target_prompt = self.prompt
        self.runtime.target_token = self.answer_token
        self.pending = None
        return {
            "module": name,
            "relative_delta": 0.01,
            "budget_total": self.budget_total,
        }

    def rollback(self):
        self.edited = []
        self.pending = None
        self.budget_total = 0.0
        self.runtime.learned = False
        self.runtime.panel_shift = False
        return {"restored": []}


class TestCandidateLayers(unittest.TestCase):
    def test_defaults_include_measured_regions(self):
        layers = candidate_layers(12)
        self.assertIn(8, layers)
        self.assertTrue(all(0 <= layer < 12 for layer in layers))
        self.assertEqual(layers, sorted(set(layers)))

    def test_small_models(self):
        self.assertEqual(candidate_layers(1), [0])
        self.assertLessEqual(len(candidate_layers(4, count=20)), 4)

    def test_invalid_windows(self):
        with self.assertRaises(ValueError):
            candidate_layers(0)
        with self.assertRaises(ValueError):
            candidate_layers(8, low=0.9, high=0.5)


class TestTeachPolicy(unittest.TestCase):
    def test_defaults(self):
        policy = TeachPolicy()
        self.assertEqual(policy.max_rel_norm, 0.02)
        self.assertEqual(policy.min_panel_agreement, 0.875)
        self.assertIn("max_kl_p95", policy.to_dict())

    def test_invalid_values(self):
        with self.assertRaises(ValueError):
            TeachPolicy(max_rel_norm=0)
        with self.assertRaises(ValueError):
            TeachPolicy(max_kl_p95=0)
        with self.assertRaises(ValueError):
            TeachPolicy(max_ppl_ratio=0.5)
        with self.assertRaises(ValueError):
            TeachPolicy(min_panel_agreement=1.5)


class TestTeachFact(unittest.TestCase):
    def test_sentence_wrapper_compiles_the_intent(self):
        session = FakeSession(base_correct=True)
        result = teach(session, "The capital is -> Paris")
        self.assertEqual(result.status, "already_true")
        self.assertEqual(result.prompt, "The capital is")

    def test_explicit_answer_skips_the_compiler(self):
        session = FakeSession(base_correct=True)
        result = teach(session, "The capital is", " Paris")
        self.assertEqual(result.status, "already_true")
        self.assertIn("Paris", result.answer)

    def test_already_true_makes_no_edit(self):
        session = FakeSession(base_correct=True)
        result = teach_fact(session, "The capital is", " Paris")
        self.assertEqual(result.status, "already_true")
        self.assertEqual(session.search_calls, 0)
        self.assertEqual(session.edited, [])
        self.assertIn("already", result.summary)
        self.assertTrue(result.accepted)

    def test_applied_edit_passes_local_checks(self):
        session = FakeSession(winning=(5,))
        result = teach_fact(session, "The capital is", " Paris")
        self.assertEqual(result.status, "applied")
        self.assertTrue(result.applied)
        self.assertFalse(result.rolled_back)
        self.assertEqual(result.best_layer, 5)
        self.assertEqual(session.edited,
                         ["model.layers.5.mlp.down_proj.weight"])
        checks = {check["name"]: check for check in result.checks}
        self.assertTrue(checks["target"]["passed"])
        self.assertTrue(checks["budget"]["passed"])
        self.assertTrue(checks["panel_top1"]["passed"])
        self.assertTrue(checks["kl_p95"]["passed"])
        self.assertIn("changed", result.summary)
        self.assertTrue(result.trials)

    def test_panel_shift_rejects_and_rolls_back(self):
        session = FakeSession(winning=(5,), panel_shift=True)
        result = teach_fact(session, "The capital is", " Paris")
        self.assertEqual(result.status, "rejected")
        self.assertTrue(result.rolled_back)
        self.assertEqual(session.edited, [])
        self.assertFalse(result.applied)
        self.assertIn("panel_top1", result.summary)

    def test_high_kl_rejects_and_rolls_back(self):
        session = FakeSession(winning=(5,), kl=0.9)
        result = teach_fact(session, "The capital is", " Paris")
        self.assertEqual(result.status, "rejected")
        self.assertTrue(result.rolled_back)
        self.assertIn("kl_p95", result.summary)

    def test_no_layer_hit_is_rejected_without_writing(self):
        session = FakeSession(winning=())
        result = teach_fact(session, "The capital is", " Paris")
        self.assertEqual(result.status, "rejected")
        self.assertFalse(result.rolled_back)
        self.assertEqual(session.edited, [])
        self.assertIn("no candidate layer", result.summary)

    def test_dry_run_does_not_apply(self):
        session = FakeSession(winning=(5,))
        result = teach_fact(session, "The capital is", " Paris", dry_run=True)
        self.assertEqual(result.status, "dry_run")
        self.assertTrue(result.dry_run)
        self.assertEqual(session.edited, [])
        self.assertEqual(result.best_layer, 5)
        self.assertIn("Dry run", result.summary)

    def test_multi_token_answer_is_reported(self):
        session = FakeSession(winning=(5,))
        result = teach_fact(session, "The city is", "New York")
        self.assertEqual(result.status, "applied")
        self.assertTrue(
            any("only its first token" in warning for warning in result.warnings)
        )

    def test_leading_space_is_accepted(self):
        session = FakeSession(winning=(5,))
        result = teach_fact(session, "The capital is", "Paris")
        self.assertTrue(result.answer_text)
        self.assertEqual(result.answer_token_count, 1)

    def test_clean_session_is_required(self):
        session = FakeSession()
        session.edited = ["model.layers.0.mlp.down_proj.weight"]
        with self.assertRaises(RuntimeError):
            teach_fact(session, "The capital is", " Paris")

    def test_explicit_layers_are_validated(self):
        session = FakeSession()
        with self.assertRaises(ValueError):
            teach_fact(session, "The capital is", " Paris", layers=(0, 0))
        with self.assertRaises(ValueError):
            teach_fact(session, "The capital is", " Paris", layers=(99,))

    def test_evaluation_failure_after_apply_rolls_back(self):
        class ExplodingSession(FakeSession):
            def evaluate(self, prompts, targets):
                if self.runtime.learned:
                    raise RuntimeError("evaluation failed")
                return super().evaluate(prompts, targets)

        session = ExplodingSession(winning=(5,))
        with self.assertRaises(RuntimeError):
            teach_fact(session, "The capital is", " Paris")
        self.assertEqual(session.edited, [])
        self.assertFalse(session.runtime.learned)

    def test_result_is_json_safe(self):
        session = FakeSession(winning=(5,))
        result = teach_fact(session, "The capital is", " Paris")
        payload = result.to_dict()
        import json
        json.dumps(payload)
        self.assertEqual(payload["status"], "applied")
        self.assertIsInstance(payload["layers"], list)


class WordTokenizer:
    """A fixed, closed vocabulary so token ids round-trip exactly."""

    pad_token_id = 0
    eos_token_id = 0

    def __init__(self, size=96):
        self.words = {"tok%d" % index: index for index in range(size)}
        self.words["The capital is"] = 5

    def __call__(self, text, add_special_tokens=True):
        key = text.strip()
        return {"input_ids": [self.words.get(key, 7)]}

    def convert_ids_to_tokens(self, token):
        for word, index in reversed(list(self.words.items())):
            if index == int(token) and word.startswith("tok"):
                return word
        return "tok%d" % int(token)


@requires_hf
class TestTeachFactAgainstARealRuntime(unittest.TestCase):
    def test_already_true_is_a_no_op_on_a_real_session(self):
        torch.manual_seed(0)
        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=96, hidden_size=64, intermediate_size=128,
            num_hidden_layers=3, num_attention_heads=4,
            num_key_value_heads=4, max_position_embeddings=64,
        )).eval()
        runtime = HFModel(model)
        runtime.tokenizer = WordTokenizer()
        session = Session(runtime, "toy", EditBudget(max_total=1.0))

        prediction = int(np.argmax(runtime.next_logits(["The capital is"])[0]))
        answer = runtime.tokenizer.convert_ids_to_tokens(prediction)
        result = teach_fact(session, "The capital is", answer, layers=[1])

        self.assertEqual(result.status, "already_true")
        self.assertEqual(session.state().edited, [])
        self.assertFalse(session.state().has_baseline)
        self.assertTrue(result.accepted)


if __name__ == "__main__":
    unittest.main()
