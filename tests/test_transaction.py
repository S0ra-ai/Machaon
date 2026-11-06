import unittest

import numpy as np

from machaon.hygieia.gates import Verdict
from machaon.transaction import Workspace


class TestTransaction(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.ws = Workspace()
        self.ws.register("layer.0.down_proj", rng.normal(size=(8, 6)))
        self.original = self.ws.weight("layer.0.down_proj").copy()

    def test_rollback_is_bit_exact(self):
        txn = self.ws.begin("test")
        txn.apply("layer.0.down_proj", np.ones((8, 6)))
        txn.rollback()
        np.testing.assert_array_equal(self.ws.weight("layer.0.down_proj"), self.original)

    def test_commit_keeps_change(self):
        txn = self.ws.begin("test")
        txn.apply("layer.0.down_proj", np.ones((8, 6)))
        self.assertTrue(txn.commit())
        np.testing.assert_allclose(
            self.ws.weight("layer.0.down_proj"), self.original + 1.0)
        self.assertEqual(len(self.ws.history), 1)

    def test_failed_verdict_rolls_back(self):
        txn = self.ws.begin("test")
        txn.apply("layer.0.down_proj", np.ones((8, 6)))
        self.assertFalse(txn.commit(Verdict(passed=False, failures=["kl_p95"])))
        np.testing.assert_array_equal(self.ws.weight("layer.0.down_proj"), self.original)

    def test_passed_verdict_commits(self):
        txn = self.ws.begin("test")
        txn.apply("layer.0.down_proj", np.ones((8, 6)))
        self.assertTrue(txn.commit(Verdict(passed=True)))

    def test_shape_mismatch_rejected(self):
        txn = self.ws.begin("test")
        with self.assertRaises(ValueError):
            txn.apply("layer.0.down_proj", np.ones((3, 3)))

    def test_closed_transaction_cannot_reuse(self):
        txn = self.ws.begin("test")
        txn.commit()
        with self.assertRaises(RuntimeError):
            txn.apply("layer.0.down_proj", np.ones((8, 6)))

    def test_non_finite_delta_is_rejected_before_mutation(self):
        txn = self.ws.begin("test")
        bad = np.zeros((8, 6))
        bad[1, 2] = np.inf
        with self.assertRaises(ValueError):
            txn.apply("layer.0.down_proj", bad)
        np.testing.assert_array_equal(
            self.ws.weight("layer.0.down_proj"), self.original)

    def test_non_finite_weight_is_rejected_at_registration(self):
        bad = np.zeros((2, 2))
        bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.ws.register("broken", bad)


if __name__ == "__main__":
    unittest.main()
