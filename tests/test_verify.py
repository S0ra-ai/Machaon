import unittest

import numpy as np

from machaon.linalg import rank1, rank1_constrained
from machaon.verify import audit_edit, check_protected, protected_drift, relative_change


class TestVerify(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.W = rng.normal(0, 0.5, size=(32, 24))
        self.keys = rng.normal(size=(10, 24))
        self.k = rng.normal(size=24)
        self.v = self.W @ self.k + rng.normal(0, 1.5, size=32)

    def test_constrained_edit_passes(self):
        _, delta = rank1_constrained(self.W, self.k, self.v, preserve=self.keys)
        report = check_protected(self.W, delta, self.keys)
        self.assertTrue(report["passed"])
        self.assertLess(report["max_drift"], 1e-12)

    def test_naive_edit_fails(self):
        _, delta = rank1(self.W, self.k, self.v)
        report = check_protected(self.W, delta, self.keys)
        self.assertFalse(report["passed"])
        self.assertGreater(report["max_drift"], 1e-3)

    def test_drift_shapes(self):
        _, delta = rank1_constrained(self.W, self.k, self.v, preserve=self.keys)
        self.assertEqual(protected_drift(self.W, delta, self.keys).shape, (10,))

    def test_relative_change(self):
        self.assertAlmostEqual(relative_change(self.W, np.zeros_like(self.W)), 0.0)
        self.assertAlmostEqual(
            relative_change(self.W, self.W), 1.0, places=12)

    def test_audit_edit_combines_tensors(self):
        _, d1 = rank1_constrained(self.W, self.k, self.v, preserve=self.keys)
        report = audit_edit(
            {"layer0": d1}, {"layer0": self.W}, {"layer0": self.keys})
        self.assertTrue(report["passed"])
        self.assertIn("relative_change", report["tensors"]["layer0"])


if __name__ == "__main__":
    unittest.main()
