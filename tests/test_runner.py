import unittest

import numpy as np

from machaon.hygieia import EditSuite, evaluate_edit, judge


class FakeBackend:
    """Index-addressed backend: items are integers into the supplied arrays."""

    def __init__(self, logits, nll=None):
        self.logits = np.asarray(logits, dtype=np.float64)
        self._nll = None if nll is None else np.asarray(nll, dtype=np.float64)

    def next_logits(self, items):
        return self.logits[np.asarray(items)]

    def nll(self, items):
        if self._nll is None:
            raise AssertionError("nll was not supposed to be called")
        return self._nll[np.asarray(items)]


class TestEvaluateEdit(unittest.TestCase):
    def setUp(self):
        self.before = FakeBackend(
            logits=[[0.0, 5.0, 0.0], [5.0, 0.0, 0.0]],
            nll=[[1.0, 1.0], [1.0, 1.0]],
        )
        self.after = FakeBackend(
            logits=[[9.0, 0.0, 0.0], [0.0, 9.0, 0.0]],
            nll=[[1.0, 1.0], [1.0, 1.0]],
        )

    def test_efficacy_is_measured_on_targets_only(self):
        suite = EditSuite(target_items=[0, 1], target_token_ids=[0, 1])
        metrics = evaluate_edit(self.before, self.after, suite)
        self.assertEqual(metrics["rewrite_acc"], 1.0)

    def test_partial_success(self):
        suite = EditSuite(target_items=[0, 1], target_token_ids=[0, 0])
        metrics = evaluate_edit(self.before, self.after, suite)
        self.assertAlmostEqual(metrics["rewrite_acc"], 0.5)

    def test_local_metrics_are_opt_in(self):
        suite = EditSuite(target_items=[0], target_token_ids=[0])
        metrics = evaluate_edit(self.before, self.after, suite)
        self.assertNotIn("kl_p95", metrics)
        self.assertNotIn("ppl_ratio", metrics)

    def test_kl_reported_when_neutral_items_given(self):
        suite = EditSuite(target_items=[0], target_token_ids=[0], neutral_items=[0])
        metrics = evaluate_edit(self.before, self.after, suite)
        self.assertGreater(metrics["kl_p95"], 0.0)

    def test_perplexity_ratio(self):
        after = FakeBackend(logits=[[0.0]], nll=[[1.0, 2.0]])
        before = FakeBackend(logits=[[0.0]], nll=[[1.0, 1.0]])
        suite = EditSuite(target_items=[0], target_token_ids=[0], corpus_items=[0])
        metrics = evaluate_edit(before, after, suite)
        self.assertAlmostEqual(metrics["ppl_ratio"], float(np.exp(0.5)))

    def test_extra_metrics_pass_through(self):
        suite = EditSuite(target_items=[0], target_token_ids=[0], extra={"mmlu_delta": -0.003})
        metrics = evaluate_edit(self.before, self.after, suite)
        self.assertEqual(metrics["mmlu_delta"], -0.003)

    def test_empty_suite_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_edit(self.before, self.after, EditSuite([], []))

    def test_judge_is_fail_closed(self):
        suite = EditSuite(target_items=[0], target_token_ids=[0])
        verdict = judge(self.before, self.after, suite)
        self.assertFalse(verdict.passed)
        self.assertIn("mmlu_delta", verdict.missing)

    def test_judge_passes_a_narrow_gate_set(self):
        suite = EditSuite(target_items=[0], target_token_ids=[0])
        verdict = judge(self.before, self.after, suite, gates={"rewrite_acc": (">=", 0.9)})
        self.assertTrue(verdict.passed)


if __name__ == "__main__":
    unittest.main()
