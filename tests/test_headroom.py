"""Reusable headroom evaluators for WSP decisions."""
import unittest

import numpy as np

from machaon.headroom import (
    DecodingConfig,
    generation_accuracy,
    make_generation_headroom,
    make_next_token_headroom,
    next_token_accuracy,
)


class StubRuntime:
    def __init__(self, generated):
        self.generated = list(generated)
        self.calls = []

    def generate(self, prompts, **kwargs):
        self.calls.append((list(prompts), dict(kwargs)))
        return self.generated


class StubRecordRuntime(StubRuntime):
    def __init__(self, records):
        super().__init__([])
        self.records = list(records)

    def generate_records(self, prompts, **kwargs):
        self.calls.append((list(prompts), dict(kwargs)))
        return self.records


class TestDecodingConfig(unittest.TestCase):
    def test_defaults_and_kwargs(self):
        config = DecodingConfig()
        self.assertFalse(config.do_sample)
        self.assertEqual(config.kwargs()["max_new_tokens"], 128)

    def test_sampling_settings_are_forwarded(self):
        config = DecodingConfig(
            max_new_tokens=8, do_sample=True, temperature=0.7,
            top_p=0.9, top_k=20, repetition_penalty=1.1, seed=3,
        )
        kwargs = config.kwargs()
        self.assertTrue(kwargs["do_sample"])
        self.assertEqual(kwargs["temperature"], 0.7)
        self.assertEqual(kwargs["top_k"], 20)
        self.assertEqual(kwargs["seed"], 3)

    def test_invalid_values_are_refused(self):
        with self.assertRaises(ValueError):
            DecodingConfig(max_new_tokens=0)
        with self.assertRaises(ValueError):
            DecodingConfig(top_p=0)
        with self.assertRaises(ValueError):
            DecodingConfig(top_k=-1)


class TestGenerationAccuracy(unittest.TestCase):
    def test_exact_match_and_samples(self):
        runtime = StubRuntime([" Rome", "Paris", "Lyon"])
        report = generation_accuracy(
            runtime, ["a", "b", "c"], ["Rome", "Paris", "Nice"],
            include_samples=True,
        )
        self.assertEqual(report["accuracy"], 2 / 3)
        self.assertEqual(report["n"], 3)
        self.assertIn("decoding", report)
        self.assertEqual(len(report["samples"]), 3)

    def test_contains_and_prefix(self):
        runtime = StubRuntime(["the answer is Rome", "Paris is nice"])
        contains = generation_accuracy(
            runtime, ["a", "b"], ["Rome", "Paris"], match="contains"
        )
        self.assertEqual(contains["accuracy"], 1.0)
        prefix = generation_accuracy(
            runtime, ["a", "b"], ["the answer", "Paris"], match="prefix"
        )
        self.assertEqual(prefix["accuracy"], 1.0)

    def test_bad_match_is_refused(self):
        with self.assertRaises(ValueError):
            generation_accuracy(StubRuntime(["x"]), ["a"], ["x"], match="regex")

    def test_runner_sets_name(self):
        runner = make_generation_headroom(
            ["a"], ["Rome"], DecodingConfig(max_new_tokens=4), name="sampled"
        )
        report = runner(StubRuntime(["Rome"]))
        self.assertEqual(report["name"], "sampled")
        self.assertEqual(report["accuracy"], 1.0)

    def test_mechanism_metadata_is_included_when_requested(self):
        runtime = StubRecordRuntime([
            {"text": "Rome", "generated_tokens": 2, "hit_token_limit": False},
        ])
        report = generation_accuracy(
            runtime, ["a"], ["Rome"], include_mechanism=True
        )
        self.assertEqual(report["mechanism"]["truncation_rate"], 0.0)
        self.assertEqual(report["mechanism"]["n"], 1)


class StubTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [3 if text.strip().lower() == "rome" else 4]}


class StubLogitRuntime:
    tokenizer = StubTokenizer()

    def next_logits(self, prompts):
        return np.asarray([[0.0, 0.0, 0.0, 5.0, 1.0]])


class TestNextTokenAccuracy(unittest.TestCase):
    def test_accuracy_and_runner(self):
        runtime = StubLogitRuntime()
        report = next_token_accuracy(runtime, ["p"], ["Rome"])
        self.assertEqual(report["accuracy"], 1.0)
        runner = make_next_token_headroom(["p"], ["Rome"], name="first_token")
        self.assertEqual(runner(runtime)["name"], "first_token")

    def test_length_mismatch_is_refused(self):
        with self.assertRaises(ValueError):
            next_token_accuracy(StubLogitRuntime(), ["p"], ["Rome", "Paris"])


if __name__ == "__main__":
    unittest.main()
