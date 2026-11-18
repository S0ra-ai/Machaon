"""Batch least-squares edits share one delta, one budget and one audit."""
import unittest

import numpy as np

from machaon.batch import solve_batch
from machaon.linalg import relative_norm


class TestBatchSolve(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        self.W = rng.normal(0.0, 0.5, size=(8, 6))
        self.keys = rng.normal(0.0, 1.0, size=(3, 6))
        self.values = rng.normal(0.0, 1.0, size=(3, 8))

    def test_exact_batch_hits_every_key(self):
        solution = solve_batch(self.W, self.keys, self.values)
        np.testing.assert_allclose(
            (self.W + solution.delta) @ self.keys.T,
            self.values.T,
            atol=1e-10,
        )
        self.assertTrue(solution.exact)
        self.assertFalse(solution.clamped)
        self.assertLess(solution.target_error, 1e-10)
        self.assertLessEqual(solution.rank, len(self.keys))

    def test_protected_keys_are_invariant_when_targets_are_representable(self):
        preserve = np.random.default_rng(12).normal(size=(2, 6))
        solution = solve_batch(
            self.W, self.keys, self.values, preserve=preserve
        )
        np.testing.assert_allclose(
            solution.delta @ preserve.T, np.zeros((8, 2)), atol=1e-10
        )
        self.assertTrue(solution.exact)

    def test_rank_deficient_protected_target_is_reported_not_hidden(self):
        # Three target rows projected away from a 2-D protected subspace leave
        # only two dimensions in this 4-D weight, so one target cannot be met.
        W = np.random.default_rng(13).normal(size=(5, 4))
        keys = np.random.default_rng(14).normal(size=(3, 4))
        values = np.random.default_rng(15).normal(size=(3, 5))
        preserve = np.random.default_rng(16).normal(size=(2, 4))
        solution = solve_batch(W, keys, values, preserve=preserve)
        self.assertFalse(solution.exact)
        self.assertGreater(solution.target_error, 1e-3)
        np.testing.assert_allclose(
            solution.delta @ preserve.T, np.zeros((5, 2)), atol=1e-10
        )

    def test_clamp_is_explicit_and_optional(self):
        solution = solve_batch(
            self.W, self.keys, self.values, max_rel_norm=0.01, clamp=True
        )
        self.assertTrue(solution.clamped)
        self.assertFalse(solution.exact)
        self.assertAlmostEqual(solution.relative_norm, 0.01, places=12)
        with self.assertRaises(ValueError):
            solve_batch(
                self.W, self.keys, self.values,
                max_rel_norm=0.01, clamp=False,
            )

    def test_invalid_inputs_are_refused(self):
        with self.assertRaises(ValueError):
            solve_batch(self.W, np.zeros(6), self.values)
        with self.assertRaises(ValueError):
            solve_batch(self.W, self.keys, self.values[:, :2])
        with self.assertRaises(ValueError):
            solve_batch(self.W, self.keys[:2], self.values)
        with self.assertRaises(ValueError):
            solve_batch(self.W, self.keys, self.values, ridge=-1)

    def test_result_is_json_safe(self):
        solution = solve_batch(self.W, self.keys, self.values)
        payload = solution.to_dict()
        self.assertEqual(payload["delta_shape"], list(self.W.shape))
        self.assertNotIn("delta", payload)
        self.assertIsInstance(payload["per_key_error"], list)

    def test_relative_norm_matches_helper(self):
        solution = solve_batch(self.W, self.keys, self.values)
        self.assertAlmostEqual(
            solution.relative_norm,
            relative_norm(solution.delta, self.W),
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
