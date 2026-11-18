"""WSP acceptance: matched controls and repair-vs-acquisition classification."""
import unittest

import numpy as np

from machaon.wsp import (
    WSPProtocol,
    assess_report,
    assess_wsp,
    classify_effect,
    random_control,
    random_controls,
)

GOOD = {
    "rewrite_acc": 0.96,
    "paraphrase_acc": 0.80,
    "unrelated_acc": 1.00,
    "kl_p95": 0.01,
    "kl_max": 0.10,
    "mmlu_delta": 0.00,
    "ppl_ratio": 1.00,
    "longform_contradictions": 0,
}


class TestRandomControls(unittest.TestCase):
    def test_control_has_same_shape_support_and_norm(self):
        delta = np.zeros((6, 5))
        delta[1, 2] = 0.3
        delta[4, 0] = -0.4
        delta[2, 4] = 0.1
        control = random_control(delta, seed=7)
        self.assertEqual(control.shape, delta.shape)
        np.testing.assert_array_equal(control != 0, delta != 0)
        self.assertAlmostEqual(
            np.linalg.norm(control), np.linalg.norm(delta), places=12
        )

    def test_control_is_deterministic(self):
        delta = np.arange(12, dtype=np.float64).reshape(3, 4)
        a = random_control(delta, seed=3)
        b = random_control(delta, seed=3)
        c = random_control(delta, seed=4)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))

    def test_zero_delta_stays_zero(self):
        np.testing.assert_array_equal(
            random_control(np.zeros((2, 3))), np.zeros((2, 3))
        )

    def test_dense_control_breaks_support(self):
        delta = np.zeros((3, 3))
        delta[1, 1] = 1.0
        control = random_control(delta, seed=1, preserve_support=False)
        self.assertGreater(int((control != 0).sum()), 1)

    def test_per_tensor_controls_are_stable(self):
        deltas = {"a": np.ones((2, 2)), "b": np.ones((3, 1))}
        first = random_controls(deltas, seed=11)
        second = random_controls(deltas, seed=11)
        for name in deltas:
            np.testing.assert_array_equal(first[name], second[name])
        self.assertNotEqual(
            first["a"].tobytes(), first["b"].tobytes()
        )

    def test_invalid_shape_is_refused(self):
        with self.assertRaises(ValueError):
            random_control(np.ones(3))

    def test_non_finite_delta_is_refused(self):
        delta = np.zeros((2, 2))
        delta[0, 0] = np.nan
        with self.assertRaises(ValueError):
            random_control(delta)


class TestClassification(unittest.TestCase):
    def test_repair_when_candidate_reaches_the_headroom(self):
        verdict = classify_effect(
            {"rewrite_acc": 0.20},
            {"rewrite_acc": 0.96},
            controls={"random": {"rewrite_acc": 0.20}},
            headroom={"name": "thinking_off", "rewrite_acc": 0.94},
        )
        self.assertEqual(verdict.label, "repair")
        self.assertTrue(verdict.accepted)
        self.assertGreater(verdict.headroom_gap, 0.0)

    def test_acquisition_when_candidate_exceeds_headroom_and_controls(self):
        verdict = classify_effect(
            {"rewrite_acc": 0.20},
            {"rewrite_acc": 0.96},
            controls={"random": {"rewrite_acc": 0.20}},
            headroom={"name": "thinking_off", "rewrite_acc": 0.50},
        )
        self.assertEqual(verdict.label, "acquisition")
        self.assertTrue(verdict.accepted)

    def test_non_specific_when_control_matches_candidate(self):
        verdict = classify_effect(
            {"rewrite_acc": 0.20},
            {"rewrite_acc": 0.96},
            controls={"random": {"rewrite_acc": 0.94}},
            headroom={"rewrite_acc": 0.50},
        )
        self.assertEqual(verdict.label, "non_specific")
        self.assertFalse(verdict.accepted)

    def test_no_effect_below_threshold(self):
        verdict = classify_effect(
            {"rewrite_acc": 0.20},
            {"rewrite_acc": 0.22},
            controls={"random": {"rewrite_acc": 0.20}},
        )
        self.assertEqual(verdict.label, "no_effect")
        self.assertFalse(verdict.accepted)

    def test_regression(self):
        verdict = classify_effect(
            {"rewrite_acc": 0.50},
            {"rewrite_acc": 0.20},
            controls={"random": {"rewrite_acc": 0.50}},
        )
        self.assertEqual(verdict.label, "regression")
        self.assertFalse(verdict.accepted)

    def test_positive_without_headroom_is_unclassified(self):
        verdict = classify_effect(
            {"rewrite_acc": 0.20},
            {"rewrite_acc": 0.96},
            controls={"random": {"rewrite_acc": 0.20}},
        )
        self.assertEqual(verdict.label, "unclassified")
        self.assertFalse(verdict.accepted)

    def test_missing_controls_is_inconclusive(self):
        verdict = classify_effect(
            {"rewrite_acc": 0.20},
            {"rewrite_acc": 0.96},
            headroom={"rewrite_acc": 0.94},
        )
        self.assertEqual(verdict.label, "inconclusive")
        self.assertFalse(verdict.accepted)

    def test_missing_metric_is_a_clear_error(self):
        with self.assertRaises(KeyError):
            classify_effect({}, {"rewrite_acc": 0.96})

    def test_protocol_thresholds_are_validated(self):
        with self.assertRaises(ValueError):
            WSPProtocol(min_gain=-0.1)
        with self.assertRaises(ValueError):
            WSPProtocol(target_metric="")


class TestAssessment(unittest.TestCase):
    def test_candidate_gate_failure_keeps_the_effect_but_rejects(self):
        candidate = dict(GOOD)
        del candidate["kl_p95"]
        report = assess_wsp(
            {"rewrite_acc": 0.20},
            candidate,
            controls={"random": {"rewrite_acc": 0.20}},
            headroom={"rewrite_acc": 0.94},
        )
        self.assertEqual(report["label"], "repair")
        self.assertFalse(report["accepted"])
        self.assertIn("kl_p95", report["candidate_gate"]["missing"])

    def test_gate_passing_repair_is_accepted(self):
        report = assess_wsp(
            {"rewrite_acc": 0.20},
            GOOD,
            controls={"random": {"rewrite_acc": 0.20}},
            headroom={"name": "thinking_off", "rewrite_acc": 0.94},
        )
        self.assertTrue(report["accepted"])
        self.assertEqual(report["label"], "repair")
        self.assertTrue(report["candidate_gate"]["passed"])
        self.assertFalse(report["control_gates"]["random"]["passed"])

    def test_report_parser_roundtrip(self):
        payload = {
            "base": {"rewrite_acc": 0.20},
            "candidate": dict(GOOD),
            "controls": {"random": {"rewrite_acc": 0.20}},
            "headroom": {"name": "thinking_off", "rewrite_acc": 0.94},
        }
        report = assess_report(payload)
        self.assertTrue(report["accepted"])
        self.assertEqual(report["label"], "repair")
        self.assertEqual(report["protocol"]["target_metric"], "rewrite_acc")

    def test_report_parser_refuses_missing_base(self):
        with self.assertRaises(ValueError):
            assess_report({"candidate": {"rewrite_acc": 0.9}})

    def test_report_parser_refuses_non_mapping_control(self):
        with self.assertRaises(ValueError):
            assess_report({
                "base": {"rewrite_acc": 0.2},
                "candidate": {"rewrite_acc": 0.9},
                "controls": {"random": 0.2},
            })

    def test_mechanism_gate_can_reject_an_otherwise_accepted_repair(self):
        report = assess_wsp(
            {"rewrite_acc": 0.20},
            GOOD,
            controls={"random": {"rewrite_acc": 0.20}},
            headroom={"name": "thinking_off", "rewrite_acc": 0.94},
            mechanism={
                "base": {"loop_rate": 0.90},
                "candidate": {"loop_rate": 0.89},
                "controls": {"random": {"loop_rate": 0.95}},
                "protocol": {
                    "metrics": ["loop_rate"],
                    "directions": {"loop_rate": "lower"},
                    "min_improvement": 0.05,
                    "control_margin": 0.05,
                },
            },
        )
        self.assertEqual(report["label"], "repair")
        self.assertFalse(report["accepted"])
        self.assertEqual(report["mechanism"]["label"], "no_effect")
        self.assertIn("mechanism gate failed", report["reason"])

    def test_mechanism_gate_passes_and_is_retained(self):
        report = assess_wsp(
            {"rewrite_acc": 0.20},
            GOOD,
            controls={"random": {"rewrite_acc": 0.20}},
            headroom={"name": "thinking_off", "rewrite_acc": 0.94},
            mechanism={
                "base": {"loop_rate": 0.90},
                "candidate": {"loop_rate": 0.30},
                "controls": {"random": {"loop_rate": 0.85}},
                "protocol": {
                    "metrics": ["loop_rate"],
                    "directions": {"loop_rate": "lower"},
                },
            },
        )
        self.assertTrue(report["accepted"])
        self.assertTrue(report["mechanism"]["passed"])
        self.assertEqual(report["mechanism"]["label"], "obstruction_relieved")


if __name__ == "__main__":
    unittest.main()
