"""Task-vector extraction and coefficient combination."""
import unittest

import numpy as np

from machaon.task_arithmetic import (
    TaskVector,
    combine_task_vectors,
    task_vector,
)


class TestTaskVectors(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4)
        self.base = rng.normal(size=(4, 3))
        self.tuned = self.base + 0.1 * rng.normal(size=self.base.shape)

    def test_extract_and_measure(self):
        vector = task_vector(self.base, self.tuned)
        np.testing.assert_array_equal(vector.delta, self.tuned - self.base)
        self.assertAlmostEqual(
            vector.relative_norm,
            np.linalg.norm(self.tuned - self.base) / np.linalg.norm(self.base),
        )
        self.assertEqual(vector.base_norm, np.linalg.norm(self.base))
        self.assertNotIn("delta", vector.to_dict())

    def test_combine_with_signed_coefficients(self):
        first = task_vector(self.base, self.tuned)
        second = TaskVector(
            delta=np.ones_like(self.base) * 0.02,
            relative_norm=0.0,
            base_norm=1.0,
        )
        combined = combine_task_vectors(
            {"a": first, "b": second},
            {"a": 2.0, "b": -0.5},
            base=self.base,
        )
        np.testing.assert_array_equal(
            combined.delta, 2.0 * first.delta - 0.5 * second.delta
        )
        self.assertEqual(combined.terms, {"a": 2.0, "b": -0.5})
        self.assertFalse(combined.clamped)
        self.assertIsNotNone(combined.relative_norm)

    def test_budget_clamps_or_refuses(self):
        first = task_vector(self.base, self.tuned)
        clamped = combine_task_vectors(
            {"a": first}, base=self.base, max_rel_norm=0.001, clamp=True
        )
        self.assertTrue(clamped.clamped)
        self.assertEqual(clamped.relative_norm, 0.001)
        with self.assertRaises(ValueError):
            combine_task_vectors(
                {"a": first}, base=self.base, max_rel_norm=0.001, clamp=False
            )

    def test_raw_arrays_are_supported(self):
        raw = np.ones_like(self.base)
        combined = combine_task_vectors({"raw": raw}, base=self.base)
        np.testing.assert_array_equal(combined.delta, raw)

    def test_missing_base_prevents_budget_checks(self):
        first = task_vector(self.base, self.tuned)
        combined = combine_task_vectors({"a": first})
        self.assertIsNone(combined.relative_norm)
        with self.assertRaises(ValueError):
            combine_task_vectors({"a": first}, max_rel_norm=0.01)

    def test_shape_mismatch_is_refused(self):
        with self.assertRaises(ValueError):
            combine_task_vectors(
                {"a": np.zeros((2, 2)), "b": np.zeros((2, 3))},
                base=np.zeros((2, 2)),
            )


if __name__ == "__main__":
    unittest.main()
