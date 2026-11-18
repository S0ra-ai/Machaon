"""Evidence-driven planning: edit, diagnose, or refuse a weight patch."""
import unittest

from machaon.planning import (
    DiagnosisProtocol,
    diagnose_and_plan,
    diagnose_failure,
    plan_surgery,
    scores_from_patch_report,
    search_layers,
)


class TestPlanning(unittest.TestCase):
    def test_capability_gap_does_not_become_a_weight_edit(self):
        plan = plan_surgery("capability_gap", layer_scores={3: 0.9})
        self.assertEqual(plan.route, "do_not_edit")
        self.assertIsNone(plan.layer)
        self.assertIn("training", plan.next_probe)

    def test_unknown_failure_requires_diagnosis(self):
        plan = plan_surgery("unknown", layer_scores={3: 0.9})
        self.assertEqual(plan.route, "diagnose")
        self.assertIn("causal patching", plan.next_probe)

    def test_obstruction_without_site_evidence_requires_diagnosis(self):
        plan = plan_surgery("obstruction")
        self.assertEqual(plan.route, "diagnose")
        self.assertIsNone(plan.layer)

    def test_obstruction_with_site_evidence_selects_the_best_layer(self):
        plan = plan_surgery(
            "obstruction",
            layer_scores={14: 0.20, 12: 0.31, 9: 0.08},
            max_rel_norm=0.015,
        )
        self.assertEqual(plan.route, "edit")
        self.assertEqual(plan.layer, 12)
        self.assertEqual(plan.max_rel_norm, 0.015)
        self.assertEqual([layer for layer, _ in plan.ranked_layers], [12, 14, 9])

    def test_site_below_threshold_requires_diagnosis(self):
        plan = plan_surgery(
            "obstruction", layer_scores={2: 0.01}, min_score=0.1
        )
        self.assertEqual(plan.route, "diagnose")
        self.assertIsNone(plan.layer)

    def test_patch_report_is_an_accepted_site_source(self):
        report = {
            "scores": {
                0: {"recovery": 0.1},
                1: {"recovery": 0.4},
                2: {"recovery": 0.2},
            }
        }
        self.assertEqual(
            scores_from_patch_report(report), {0: 0.1, 1: 0.4, 2: 0.2}
        )
        plan = plan_surgery("obstruction", patch_report=report)
        self.assertEqual(plan.layer, 1)

    def test_mixed_site_sources_are_refused(self):
        with self.assertRaises(ValueError):
            plan_surgery(
                "obstruction",
                layer_scores={1: 0.2},
                patch_report={"scores": {1: {"recovery": 0.2}}},
            )

    def test_unknown_failure_type_is_refused(self):
        with self.assertRaises(ValueError):
            plan_surgery("mystery")

    def test_plan_is_serialisable(self):
        plan = plan_surgery(
            "obstruction",
            layer_scores=[{"layer": 3, "score": 0.2}],
            metadata={"source": "unit"},
        )
        payload = plan.to_dict()
        self.assertEqual(payload["route"], "edit")
        self.assertEqual(payload["ranked_layers"][0]["layer"], 3)
        self.assertEqual(payload["metadata"]["source"], "unit")


class TestLayerSearch(unittest.TestCase):
    def test_ranks_successful_layers_and_keeps_failures(self):
        cleaned = []

        def evaluate(layer):
            if layer == 2:
                raise RuntimeError("boom")
            return {"accuracy": {1: 0.2, 3: 0.8, 4: 0.5}[layer]}

        report = search_layers([1, 2, 3, 4], evaluate, cleanup=lambda: cleaned.append(1))
        self.assertTrue(report["passed"])
        self.assertEqual(report["best_layer"], 3)
        self.assertEqual([row["layer"] for row in report["ranking"]], [3, 4, 1])
        self.assertEqual(len(cleaned), 4)
        self.assertEqual(
            next(row for row in report["trials"] if row["layer"] == 2)["status"],
            "error",
        )

    def test_lower_is_better_and_max_evaluations(self):
        report = search_layers(
            [0, 1, 2],
            lambda layer: {"loss": {0: 0.4, 1: 0.2, 2: 0.3}[layer]},
            metric="loss",
            higher_is_better=False,
            max_evaluations=2,
        )
        self.assertEqual(report["best_layer"], 1)
        self.assertEqual([row["layer"] for row in report["trials"]], [0, 1])

    def test_cleanup_failure_aborts_the_search(self):
        calls = []

        def cleanup():
            calls.append(1)
            raise OSError("rollback failed")

        with self.assertRaises(OSError):
            search_layers([0, 1], lambda layer: {"accuracy": 1.0}, cleanup=cleanup)
        self.assertEqual(len(calls), 1)

    def test_missing_metric_records_a_failed_trial(self):
        report = search_layers([0], lambda layer: {"other": 1.0})
        self.assertFalse(report["passed"])
        self.assertIsNone(report["best_layer"])
        self.assertIn("missing ranking metric", report["trials"][0]["error"])

    def test_duplicate_layers_are_refused(self):
        with self.assertRaises(ValueError):
            search_layers([1, 1], lambda layer: {"accuracy": 1.0})


class TestDiagnosis(unittest.TestCase):
    def test_obstruction_requires_target_and_mechanism_relief(self):
        diagnosis = diagnose_failure(
            {"accuracy": 0.2},
            headroom={"accuracy": 0.9},
            mechanism={
                "base": {"loop_rate": 0.9, "truncation_rate": 0.9},
                "headroom": {"loop_rate": 0.3, "truncation_rate": 0.3},
            },
        )
        self.assertEqual(diagnosis.failure_type, "obstruction")
        self.assertEqual(diagnosis.confidence, "high")

    def test_target_headroom_without_mechanism_is_unknown(self):
        diagnosis = diagnose_failure(
            {"accuracy": 0.2}, headroom={"accuracy": 0.9}
        )
        self.assertEqual(diagnosis.failure_type, "unknown")
        self.assertIn("mechanism", diagnosis.reason)

    def test_low_headroom_without_mechanism_relief_is_capability_gap(self):
        diagnosis = diagnose_failure(
            {"accuracy": 0.2},
            headroom={"accuracy": 0.21},
            mechanism={
                "base": {"loop_rate": 0.2},
                "headroom": {"loop_rate": 0.2},
            },
        )
        self.assertEqual(diagnosis.failure_type, "capability_gap")
        self.assertIn("data", diagnosis.next_step)

    def test_missing_headroom_is_unknown(self):
        diagnosis = diagnose_failure({"accuracy": 0.2})
        self.assertEqual(diagnosis.failure_type, "unknown")

    def test_lower_headroom_is_refused_as_unknown(self):
        diagnosis = diagnose_failure(
            {"accuracy": 0.9}, headroom={"accuracy": 0.2}
        )
        self.assertEqual(diagnosis.failure_type, "unknown")
        self.assertIn("below the base", diagnosis.reason)

    def test_custom_protocol_and_plan_integration(self):
        protocol = DiagnosisProtocol(
            target_metric="score", min_target_gap=0.1, min_mechanism_gap=0.1
        )
        result = diagnose_and_plan(
            {"score": 0.2},
            headroom={"score": 0.7},
            mechanism={
                "base": {"loop_rate": 0.8},
                "headroom": {"loop_rate": 0.2},
            },
            protocol=protocol,
            plan_kwargs={"layer_scores": {12: 0.3}},
        )
        self.assertEqual(result["diagnosis"]["failure_type"], "obstruction")
        self.assertEqual(result["plan"]["route"], "edit")
        self.assertEqual(result["plan"]["layer"], 12)


if __name__ == "__main__":
    unittest.main()
