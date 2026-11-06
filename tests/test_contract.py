"""Budget-aware contract synthesis keeps every admitted guarantee exact."""
from __future__ import annotations

import unittest

import numpy as np

from machaon.budget import BudgetExceeded
from machaon.contract import synthesize_contract
from machaon.linalg import rank1, rank1_constrained, relative_norm
from machaon.verify import check_protected


class TestContractSynthesis(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.W = rng.normal(0.0, 0.5, size=(24, 16))
        self.keys = rng.normal(0.0, 1.0, size=(12, 16))
        self.k = rng.normal(0.0, 1.0, size=16)
        self.v = self.W @ self.k + rng.normal(0.0, 1.5, size=24)

    def test_large_budget_admits_every_key(self):
        _, base = rank1(self.W, self.k, self.v)
        _, full = rank1_constrained(
            self.W, self.k, self.v, preserve=self.keys)
        budget = relative_norm(full, self.W) * 1.01
        solution = synthesize_contract(
            self.W, self.k, self.v, self.keys, budget)
        self.assertEqual(len(solution.selected), len(self.keys))
        self.assertFalse(solution.rejected)
        self.assertLess(solution.relative_norm, budget)
        self.assertLess(solution.target_error, 1e-12)
        check = check_protected(self.W, solution.delta, self.keys)
        self.assertTrue(check["passed"])
        self.assertLess(check["max_drift"], 1e-12)

    def test_tight_budget_admits_a_certified_subset(self):
        _, base = rank1(self.W, self.k, self.v)
        _, full = rank1_constrained(
            self.W, self.k, self.v, preserve=self.keys)
        base_rel = relative_norm(base, self.W)
        full_rel = relative_norm(full, self.W)
        self.assertGreater(full_rel, base_rel)
        budget = (base_rel + full_rel) / 2
        solution = synthesize_contract(
            self.W, self.k, self.v, self.keys, budget)
        self.assertGreater(len(solution.selected), 0)
        self.assertLess(len(solution.selected), len(self.keys))
        self.assertLessEqual(solution.relative_norm, budget)
        self.assertLess(solution.target_error, 1e-12)
        admitted = self.keys[list(solution.selected)]
        check = check_protected(self.W, solution.delta, admitted)
        self.assertTrue(check["passed"])
        self.assertLess(check["max_drift"], 1e-12)

    def test_base_update_above_budget_is_refused(self):
        _, base = rank1(self.W, self.k, self.v)
        budget = relative_norm(base, self.W) * 0.5
        with self.assertRaises(BudgetExceeded):
            synthesize_contract(
                self.W, self.k, self.v, self.keys, budget)

    def test_weights_select_at_least_one_high_priority_key(self):
        _, base = rank1(self.W, self.k, self.v)
        weights = np.arange(len(self.keys), dtype=np.float64)
        budget = relative_norm(base, self.W) * 1.02
        solution = synthesize_contract(
            self.W, self.k, self.v, self.keys, budget, weights=weights)
        self.assertTrue(solution.selected)

    def test_invalid_candidates_are_refused(self):
        with self.assertRaises(ValueError):
            synthesize_contract(
                self.W, self.k, self.v, np.zeros(16), 1.0)

    def test_negative_weights_are_refused(self):
        with self.assertRaises(ValueError):
            synthesize_contract(
                self.W, self.k, self.v, self.keys, 1.0,
                weights=np.r_[0.0, -1.0, np.zeros(10)])


if __name__ == "__main__":
    unittest.main()
