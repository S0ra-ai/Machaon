"""Mechanism evidence: loop, truncation and matched-control gates."""
import unittest

from machaon.mechanism import (
    MechanismProtocol,
    assess_mechanism,
    generation_mechanism,
    generation_metrics,
    records_metrics,
)


class TestGenerationMetrics(unittest.TestCase):
    def test_loop_empty_and_truncation_rates(self):
        report = generation_metrics(
            ["a b c d a b c d a b c d", ""],
            hit_token_limit=[True, False],
            generated_tokens=[12, 0],
        )
        self.assertEqual(report["n"], 2)
        self.assertGreater(report["loop_rate"], 0.2)
        self.assertEqual(report["empty_rate"], 0.5)
        self.assertEqual(report["truncation_rate"], 0.5)
        self.assertEqual(report["mean_generated_tokens"], 6.0)

    def test_records_without_stop_metadata_omit_truncation(self):
        report = records_metrics([{"text": "a b c d e f g"}])
        self.assertNotIn("truncation_rate", report)
        self.assertEqual(report["loop_rate"], 0.0)

    def test_length_mismatch_is_refused(self):
        with self.assertRaises(ValueError):
            generation_metrics(["a"], hit_token_limit=[True, False])


class TestMechanismAssessment(unittest.TestCase):
    def setUp(self):
        self.base = {"loop_rate": 0.90, "truncation_rate": 0.90}
        self.candidate = {"loop_rate": 0.30, "truncation_rate": 0.30}
        self.controls = {
            "random": {"loop_rate": 0.85, "truncation_rate": 0.85}
        }

    def test_obstruction_relieved(self):
        verdict = assess_mechanism(
            self.base,
            self.candidate,
            controls=self.controls,
        )
        self.assertTrue(verdict.passed)
        self.assertEqual(verdict.label, "obstruction_relieved")
        self.assertAlmostEqual(
            verdict.metrics["loop_rate"]["control_gap"], 0.55
        )

    def test_control_that_also_improves_is_non_specific(self):
        controls = {
            "random": {"loop_rate": 0.31, "truncation_rate": 0.31}
        }
        verdict = assess_mechanism(self.base, self.candidate, controls=controls)
        self.assertFalse(verdict.passed)
        self.assertEqual(verdict.label, "non_specific")

    def test_no_mechanism_effect_is_reported(self):
        candidate = {"loop_rate": 0.86, "truncation_rate": 0.86}
        controls = {
            "random": {"loop_rate": 0.95, "truncation_rate": 0.95}
        }
        verdict = assess_mechanism(self.base, candidate, controls=controls)
        self.assertFalse(verdict.passed)
        self.assertEqual(verdict.label, "no_effect")

    def test_missing_controls_are_inconclusive(self):
        verdict = assess_mechanism(self.base, self.candidate)
        self.assertFalse(verdict.passed)
        self.assertEqual(verdict.label, "inconclusive")

    def test_custom_protocol_can_select_one_metric(self):
        protocol = MechanismProtocol(
            metrics=("loop_rate",),
            directions=(("loop_rate", "lower"),),
            min_improvement=0.2,
            control_margin=0.2,
        )
        verdict = assess_mechanism(
            self.base, self.candidate, controls=self.controls, protocol=protocol
        )
        self.assertTrue(verdict.passed)
        self.assertEqual(set(verdict.metrics), {"loop_rate"})

    def test_generation_mechanism_builder(self):
        records = [{"text": "a b c d e f g", "hit_token_limit": False,
                    "generated_tokens": 7}]
        payload = generation_mechanism(records, records, controls={"c": records})
        self.assertEqual(set(payload), {"base", "candidate", "controls"})
        self.assertEqual(payload["controls"]["c"]["truncation_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
