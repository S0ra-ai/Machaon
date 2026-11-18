"""The obstruction-vs-capability benchmark harness and its fixture."""
import os
import unittest

from machaon.bench.wsp import (
    WSPBenchmarkCase,
    evaluate_case,
    load_benchmark,
    run_benchmark,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestBenchmarkCase(unittest.TestCase):
    def test_example_fixture_runs_and_matches(self):
        cases = load_benchmark(os.path.join(ROOT, "examples", "wsp_benchmark.json"))
        self.assertEqual(len(cases), 2)
        report = run_benchmark(cases)
        self.assertTrue(report["summary"]["all_matched"])
        self.assertEqual(report["summary"]["confusion"]["true_positive"], 1)
        self.assertEqual(report["summary"]["confusion"]["true_negative"], 1)
        self.assertTrue(report["summary"]["contains_only_synthetic"])
        self.assertEqual(report["summary"]["source_kinds"], {"synthetic": 2})

    def test_measured_reference_fixture_passes_provenance_gate(self):
        cases = load_benchmark(
            os.path.join(ROOT, "examples", "wsp_measured_smoke.json")
        )
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].source_kind, "measured")
        report = run_benchmark(cases)
        self.assertTrue(report["summary"]["all_matched"])
        self.assertEqual(report["summary"]["source_kinds"], {"measured": 1})
        self.assertFalse(report["summary"]["contains_only_synthetic"])

    def test_obstruction_case_is_accepted(self):
        case = WSPBenchmarkCase.from_dict({
            "case_id": "obstruction",
            "failure_type": "obstruction",
            "expected_repairable": True,
            "base": {"rewrite_acc": 0.2},
            "candidate": {"rewrite_acc": 0.96},
            "controls": {"random": {"rewrite_acc": 0.2}},
            "headroom": {"rewrite_acc": 0.94},
            "gates": {"rewrite_acc": [">=", 0.95]},
            "mechanism": {
                "base": {"loop_rate": 0.9},
                "candidate": {"loop_rate": 0.3},
                "controls": {"random": {"loop_rate": 0.85}},
                "protocol": {
                    "metrics": ["loop_rate"],
                    "directions": {"loop_rate": "lower"},
                },
            },
        })
        row = evaluate_case(case)
        self.assertTrue(row["matched"])
        self.assertTrue(row["observed_repairable"])
        self.assertEqual(row["mechanism"]["label"], "obstruction_relieved")

    def test_capability_gap_is_rejected(self):
        case = WSPBenchmarkCase.from_dict({
            "case_id": "gap",
            "failure_type": "capability_gap",
            "expected_repairable": False,
            "base": {"rewrite_acc": 0.2},
            "candidate": {"rewrite_acc": 0.22},
            "controls": {"random": {"rewrite_acc": 0.2}},
            "headroom": {"rewrite_acc": 0.2},
        })
        row = evaluate_case(case)
        self.assertTrue(row["matched"])
        self.assertFalse(row["observed_repairable"])
        self.assertEqual(row["label"], "no_effect")

    def test_unknown_failure_type_is_refused(self):
        with self.assertRaises(ValueError):
            WSPBenchmarkCase.from_dict({
                "case_id": "x",
                "failure_type": "mystery",
                "expected_repairable": True,
                "base": {"rewrite_acc": 0.2},
                "candidate": {"rewrite_acc": 0.9},
            })

    def test_duplicate_ids_are_refused(self):
        case = WSPBenchmarkCase.from_dict({
            "case_id": "same",
            "failure_type": "capability_gap",
            "expected_repairable": False,
            "base": {"rewrite_acc": 0.2},
            "candidate": {"rewrite_acc": 0.2},
        })
        with self.assertRaises(ValueError):
            run_benchmark([case, case])

    def test_case_roundtrip_is_serialisable(self):
        case = WSPBenchmarkCase.from_dict({
            "case_id": "roundtrip",
            "failure_type": "capability_gap",
            "expected_repairable": False,
            "base": {"rewrite_acc": 0.2},
            "candidate": {"rewrite_acc": 0.2},
            "metadata": {"source": "synthetic"},
        })
        again = WSPBenchmarkCase.from_dict(case.to_dict())
        self.assertEqual(again.case_id, case.case_id)
        self.assertEqual(again.metadata, case.metadata)
        self.assertEqual(again.source_kind, "synthetic")

    def test_measured_case_requires_reproducibility_fields(self):
        digest = "a" * 64
        case = WSPBenchmarkCase.from_dict({
            "case_id": "measured",
            "failure_type": "obstruction",
            "expected_repairable": True,
            "base": {"rewrite_acc": 0.2},
            "candidate": {"rewrite_acc": 0.9},
            "source_kind": "measured",
            "provenance": {
                "model": "toy",
                "checkpoint_sha256": digest,
                "dataset": "fixture",
                "dataset_sha256": digest,
                "evaluated_at": "2026-11-11T00:00:00Z",
                "seed": 0,
                "evaluation_script": "tests/test_wsp_bench.py",
            },
        })
        self.assertEqual(case.source_kind, "measured")

    def test_measured_case_rejects_unsigned_provenance(self):
        with self.assertRaises(ValueError):
            WSPBenchmarkCase.from_dict({
                "case_id": "measured",
                "failure_type": "obstruction",
                "expected_repairable": True,
                "base": {"rewrite_acc": 0.2},
                "candidate": {"rewrite_acc": 0.9},
                "source_kind": "measured",
                "provenance": {},
            })


if __name__ == "__main__":
    unittest.main()
