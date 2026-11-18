"""Deterministic weight fingerprints and pre/post edit reports."""
import unittest

import numpy as np

from machaon.fingerprint import (
    diff_tensors,
    model_fingerprint,
    tensor_fingerprint,
)


class TestTensorFingerprint(unittest.TestCase):
    def test_same_values_have_the_same_address(self):
        a = np.arange(12, dtype=np.float64).reshape(3, 4)
        b = a.copy()
        self.assertEqual(
            tensor_fingerprint(a).sha256, tensor_fingerprint(b).sha256
        )

    def test_shape_is_part_of_the_address(self):
        a = np.arange(6, dtype=np.float64)
        b = a.reshape(2, 3)
        self.assertNotEqual(
            tensor_fingerprint(a).sha256, tensor_fingerprint(b).sha256
        )

    def test_statistics_are_reported(self):
        report = tensor_fingerprint(np.array([[0.0, 1.0], [2.0, 3.0]]))
        self.assertEqual(report.shape, (2, 2))
        self.assertEqual(report.size, 4)
        self.assertEqual(report.zero_fraction, 0.25)
        self.assertGreater(report.norm, 0.0)
        self.assertEqual(report.to_dict()["shape"], [2, 2])

    def test_non_finite_values_are_refused(self):
        with self.assertRaises(ValueError):
            tensor_fingerprint(np.array([1.0, np.nan]))


class TestModelFingerprint(unittest.TestCase):
    def test_order_is_irrelevant(self):
        a = np.ones((2, 2))
        b = np.zeros((3, 1))
        first = model_fingerprint({"b": b, "a": a})
        second = model_fingerprint({"a": a, "b": b})
        self.assertEqual(first["sha256"], second["sha256"])
        self.assertEqual(first["n_tensors"], 2)
        self.assertEqual(first["parameters"], 7)

    def test_one_changed_tensor_changes_the_model_address(self):
        base = {"a": np.ones((2, 2)), "b": np.zeros((2, 2))}
        changed = {"a": np.ones((2, 2)), "b": np.ones((2, 2))}
        self.assertNotEqual(
            model_fingerprint(base)["sha256"],
            model_fingerprint(changed)["sha256"],
        )

    def test_empty_model_is_refused(self):
        with self.assertRaises(ValueError):
            model_fingerprint({})


class TestDiff(unittest.TestCase):
    def test_sparse_diff_reports_support_and_hashes(self):
        base = np.zeros((4, 3))
        changed = base.copy()
        changed[1, 2] = 2.0
        changed[3, 0] = -1.0
        report = diff_tensors(base, changed)
        self.assertTrue(report["changed"])
        self.assertEqual(report["changed_entries"], 2)
        self.assertEqual(report["support_fraction"], 2 / 12)
        self.assertEqual(report["changed_rows"], 2)
        self.assertEqual(report["changed_columns"], 2)
        self.assertEqual(report["max_abs_change"], 2.0)
        self.assertNotEqual(
            report["base"]["sha256"], report["changed_fingerprint"]["sha256"]
        )

    def test_identical_tensors_have_zero_diff(self):
        value = np.ones((3, 2))
        report = diff_tensors(value, value.copy())
        self.assertFalse(report["changed"])
        self.assertEqual(report["changed_entries"], 0)
        self.assertEqual(report["max_abs_change"], 0.0)

    def test_shape_mismatch_is_refused(self):
        with self.assertRaises(ValueError):
            diff_tensors(np.zeros((2, 2)), np.zeros((2, 3)))


if __name__ == "__main__":
    unittest.main()
