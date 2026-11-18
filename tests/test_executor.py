"""The declarative edit DSL has one explicit executor, not an implicit one."""
import unittest

import numpy as np

from machaon.executor import execute_spec
from machaon.schema import Constraints, EditSpec, Target


def spec(op, **kwargs):
    return EditSpec(
        op=op,
        target=Target("model.layers.0.mlp.down_proj", 0),
        intent="unit test",
        constraints=Constraints(max_rel_norm=kwargs.pop("max_rel_norm", 0.5)),
        **kwargs,
    )


class TestExecutor(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4)
        self.W = 2.0 * rng.normal(size=(5, 4))
        self.k = rng.normal(size=4)
        self.v = 0.1 * rng.normal(size=5)

    def test_rank1_hits_the_target(self):
        out = execute_spec(spec("rank1"), self.W, k=self.k, v=self.v)
        self.assertEqual(out.op, "rank1")
        self.assertEqual(out.delta.shape, self.W.shape)
        np.testing.assert_allclose(
            (self.W + out.delta) @ self.k, self.v, atol=1e-12
        )
        self.assertLess(out.target_error, 1e-12)

    def test_constrained_rank1_keeps_protected_keys(self):
        rng = np.random.default_rng(8)
        preserve = rng.normal(size=(2, 4))
        out = execute_spec(
            spec("rank1_constrained"), self.W, k=self.k, v=self.v, preserve=preserve
        )
        np.testing.assert_allclose(
            out.delta @ preserve.T, np.zeros((5, 2)), atol=1e-12
        )

    def test_bake_direction(self):
        ref = np.ones(4)
        direction = np.arange(5, dtype=np.float64)
        out = execute_spec(
            spec("bake_direction"), self.W, a_ref=ref, direction=direction, alpha=0.1
        )
        self.assertAlmostEqual(out.relative_norm, np.linalg.norm(out.delta) / np.linalg.norm(self.W))

    def test_ablate_output_and_input(self):
        W = np.ones((5, 4))
        W[0, :] = 0.02
        W[:, 0] = 0.02
        out_dir = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        out = execute_spec(spec("ablate_output"), W, direction=out_dir)
        np.testing.assert_allclose(
            out_dir @ (W + out.delta), 0.0, atol=1e-12
        )
        in_dir = np.array([1.0, 0.0, 0.0, 0.0])
        out = execute_spec(spec("ablate_input"), W, direction=in_dir)
        np.testing.assert_allclose(
            (W + out.delta) @ in_dir, 0.0, atol=1e-12
        )

    def test_low_rank_scale_reduces_rank(self):
        donor = 0.1 * np.random.default_rng(2).normal(size=self.W.shape)
        out = execute_spec(spec("low_rank_scale", rank=2), self.W, donor_delta=donor)
        self.assertLessEqual(np.linalg.matrix_rank(out.delta, tol=1e-8), 2)

    def test_search_solver_is_refused(self):
        with self.assertRaises(ValueError):
            execute_spec(spec("rank1", solver="search"), self.W, k=self.k, v=self.v)

    def test_missing_operand_is_refused(self):
        with self.assertRaises(ValueError):
            execute_spec(spec("rank1"), self.W, k=self.k)

    def test_budget_overflow_is_refused(self):
        with self.assertRaises(ValueError):
            execute_spec(
                spec("bake_direction", max_rel_norm=0.001),
                self.W,
                a_ref=np.ones(4),
                direction=np.ones(5),
                alpha=10.0,
            )

    def test_result_serialisation_omits_the_array(self):
        out = execute_spec(spec("rank1"), self.W, k=self.k, v=self.v)
        payload = out.to_dict()
        self.assertNotIn("delta", payload)
        self.assertEqual(payload["delta_shape"], list(self.W.shape))


if __name__ == "__main__":
    unittest.main()
