import unittest

import numpy as np

from machaon.budget import BudgetExceeded, EditBudget
from machaon.transaction import Workspace


class TestBudgetAccounting(unittest.TestCase):
    def setUp(self):
        self.budget = EditBudget(max_total=0.10)
        self.weight = np.eye(4)
        self.budget.register("w", self.weight)

    def test_zero_state(self):
        state = self.budget.state()
        self.assertEqual(state.total, 0.0)
        self.assertAlmostEqual(state.remaining, 0.10)

    def test_relative_change(self):
        delta = np.full((4, 4), 0.5)
        self.assertAlmostEqual(
            self.budget.relative("w", delta),
            np.linalg.norm(delta) / np.linalg.norm(self.weight),
        )

    def test_add_accumulates(self):
        delta = self.weight * 0.01
        self.budget.add([("w", delta)])
        first = self.budget.state().total
        self.budget.add([("w", delta)])
        self.assertAlmostEqual(self.budget.state().total, 2 * first)

    def test_exceeding_total_raises_and_records_nothing(self):
        delta = self.weight * 0.06
        self.budget.add([("w", delta)])
        before = self.budget.state().total
        with self.assertRaises(BudgetExceeded):
            self.budget.add([("w", delta)])
        self.assertAlmostEqual(self.budget.state().total, before)

    def test_per_tensor_limit(self):
        budget = EditBudget(max_total=10.0, max_per_tensor=0.01)
        budget.register("w", self.weight)
        with self.assertRaises(BudgetExceeded):
            budget.add([("w", self.weight * 0.02)])

    def test_unregistered_tensor(self):
        with self.assertRaises(KeyError):
            self.budget.add([("nope", np.ones((2, 2)))])


class TestBudgetedWorkspace(unittest.TestCase):
    def setUp(self):
        self.budget = EditBudget(max_total=0.05)
        self.ws = Workspace(budget=self.budget)
        self.ws.register("w", np.eye(8))
        self.original = self.ws.weight("w").copy()

    def test_commit_within_budget_charges(self):
        txn = self.ws.begin("small")
        txn.apply("w", np.eye(8) * 0.01)
        self.assertTrue(txn.commit())
        self.assertGreater(self.budget.state().total, 0.0)

    def test_exceeding_budget_rolls_back_and_raises(self):
        txn = self.ws.begin("too big")
        txn.apply("w", np.eye(8) * 0.5)
        with self.assertRaises(BudgetExceeded):
            txn.commit()
        np.testing.assert_array_equal(self.ws.weight("w"), self.original)
        self.assertEqual(self.budget.state().total, 0.0)

    def test_rollback_does_not_charge(self):
        txn = self.ws.begin("discard")
        txn.apply("w", np.eye(8) * 0.01)
        txn.rollback()
        self.assertEqual(self.budget.state().total, 0.0)

    def test_failed_verdict_does_not_charge(self):
        class Verdict:
            passed = False

        txn = self.ws.begin("verdict")
        txn.apply("w", np.eye(8) * 0.01)
        self.assertFalse(txn.commit(Verdict()))
        self.assertEqual(self.budget.state().total, 0.0)

    def test_no_budget_still_works(self):
        ws = Workspace()
        ws.register("w", np.eye(4))
        txn = ws.begin()
        txn.apply("w", np.ones((4, 4)))
        self.assertTrue(txn.commit())


if __name__ == "__main__":
    unittest.main()
