import unittest

import numpy as np

from machaon.hygieia.gates import evaluate
from machaon.hygieia.longform import (
    LONGFORM_GATES,
    longform_report,
    repeated_ngram_rate,
)


class StubRuntime:
    """Returns fixed generations and a fixed NLL, so metrics are checkable."""

    def __init__(self, texts, nll):
        self.texts = list(texts)
        self.nll_value = float(nll)

    def generate(self, prompts, max_new_tokens=128):
        return self.texts

    def nll(self, texts):
        return np.full(len(list(texts)), self.nll_value)


class TestRepetition(unittest.TestCase):
    def test_unique_text_has_zero_repetition(self):
        self.assertEqual(repeated_ngram_rate("a b c d e f g"), 0.0)

    def test_looping_text_is_detected(self):
        self.assertGreater(repeated_ngram_rate("a b c d a b c d a b c d"), 0.5)

    def test_short_text_returns_zero(self):
        self.assertEqual(repeated_ngram_rate("a b"), 0.0)


class TestLongformReport(unittest.TestCase):
    def test_identical_models_ratio_is_one(self):
        pristine = StubRuntime(["one two three four five"], nll=2.0)
        edited = StubRuntime(["one two three four five"], nll=2.0)
        report = longform_report(pristine, edited, ["p"], max_new_tokens=8)
        self.assertAlmostEqual(report["longform_nll_ratio"], 1.0)
        self.assertEqual(report["longform_repetition"], 0.0)
        self.assertEqual(report["longform_repetition_delta"], 0.0)
        self.assertEqual(report["longform_empty_rate"], 0.0)

    def test_damage_shows_up_as_a_ratio(self):
        class TextAware(StubRuntime):
            def nll(self, texts):
                # the reference model finds damaged text less likely
                return np.array([3.0 if "broken" in t else 2.0 for t in texts])

        pristine = TextAware(["clean text here"], nll=0.0)
        edited = TextAware(["broken text here"], nll=0.0)
        report = longform_report(pristine, edited, ["p"], max_new_tokens=8)
        self.assertAlmostEqual(report["longform_nll_ratio"], 1.5, places=6)

    def test_empty_generation_is_reported(self):
        pristine = StubRuntime(["text"], nll=1.0)
        edited = StubRuntime(["", "  "], nll=1.0)
        report = longform_report(pristine, edited, ["p"], max_new_tokens=8)
        self.assertEqual(report["longform_empty_rate"], 1.0)

    def test_no_prompts_rejected(self):
        with self.assertRaises(ValueError):
            longform_report(StubRuntime(["x"], 1.0), StubRuntime(["x"], 1.0), [])


class TestLongformGates(unittest.TestCase):
    def test_clean_report_passes(self):
        metrics = {"longform_repetition_delta": 0.01, "longform_nll_ratio": 1.02,
                   "longform_empty_rate": 0.0}
        self.assertTrue(evaluate(metrics, LONGFORM_GATES).passed)

    def test_damaged_report_fails(self):
        metrics = {"longform_repetition_delta": 0.4, "longform_nll_ratio": 1.9,
                   "longform_empty_rate": 0.0}
        verdict = evaluate(metrics, LONGFORM_GATES)
        self.assertFalse(verdict.passed)
        self.assertEqual(len(verdict.failures), 2)


if __name__ == "__main__":
    unittest.main()
