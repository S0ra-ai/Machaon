"""The offline demo and the value solvers are part of the public story."""
import contextlib
import io
import unittest

import numpy as np

from machaon.asclepius.solve import refine_value, value_from_donor
from machaon.demo import main as demo_main
from machaon.demo import run_demo


class TestDemo(unittest.TestCase):
    def test_constrained_edit_leaves_protected_keys_alone(self):
        report = run_demo(seed=1, d_in=16, d_out=16, n_preserve=8, n_probe=32)
        naive, safe, capped = report["rows"]
        self.assertGreater(naive["preserve_drift_max"], 1e-6)
        self.assertLess(safe["preserve_drift_max"], 1e-12)
        self.assertLess(capped["preserve_drift_max"], 1e-12)
        self.assertLess(safe["target_err"], 1e-12)
        self.assertLessEqual(capped["rel_delta"], report["config"]["max_rel_norm"] + 1e-12)
        self.assertGreater(capped["target_err"], safe["target_err"])

    def test_demo_main_prints_a_table(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            demo_main()
        text = out.getvalue()
        self.assertIn("rank1 (naive)", text)
        self.assertIn("constrained (capped)", text)
        self.assertIn("budget", text)


class TestSolvers(unittest.TestCase):
    def test_donor_flattens_and_copies(self):
        source = np.arange(6, dtype=np.float32).reshape(2, 3)
        value = value_from_donor(source)
        self.assertEqual(value.dtype, np.float64)
        np.testing.assert_array_equal(value, source.reshape(-1))
        source[0, 0] = 99
        self.assertEqual(value[0], 0.0)

    def test_refine_value_moves_towards_the_score(self):
        target = np.array([1.0, -1.0, 0.5])

        def score(v):
            return -float(np.linalg.norm(v - target))

        best, best_score = refine_value(
            np.zeros(3), score, sigma=0.3, steps=100, seed=0)
        self.assertGreater(best_score, score(np.zeros(3)))
        self.assertLess(np.linalg.norm(best - target), np.linalg.norm(target))

    def test_refine_value_can_minimise(self):
        best, score = refine_value(
            np.zeros(2), lambda v: float(np.sum(v * v)),
            sigma=0.2, steps=40, seed=1, maximize=False)
        self.assertLess(score, 1.0)
        self.assertEqual(best.shape, (2,))


if __name__ == "__main__":
    unittest.main()
