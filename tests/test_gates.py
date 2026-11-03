import unittest

from machaon.hygieia.gates import evaluate


GOOD = {
    "rewrite_acc": 1.0,
    "paraphrase_acc": 0.8,
    "unrelated_acc": 1.0,
    "kl_p95": 0.01,
    "kl_max": 0.1,
    "mmlu_delta": 0.0,
    "ppl_ratio": 1.0,
    "longform_contradictions": 0.0,
}


class TestGates(unittest.TestCase):
    def test_all_good_passes(self):
        self.assertTrue(evaluate(GOOD).passed)

    def test_capability_loss_fails(self):
        bad = dict(GOOD, mmlu_delta=-0.05)
        verdict = evaluate(bad)
        self.assertFalse(verdict.passed)
        self.assertTrue(any("mmlu_delta" in f for f in verdict.failures))

    def test_locality_fail(self):
        self.assertFalse(evaluate(dict(GOOD, kl_p95=0.2)).passed)

    def test_missing_metric_fails_closed(self):
        partial = dict(GOOD)
        del partial["longform_contradictions"]
        verdict = evaluate(partial)
        self.assertFalse(verdict.passed)
        self.assertIn("longform_contradictions", verdict.missing)

    def test_contradiction_is_disqualifying(self):
        self.assertFalse(evaluate(dict(GOOD, longform_contradictions=1.0)).passed)


if __name__ == "__main__":
    unittest.main()
