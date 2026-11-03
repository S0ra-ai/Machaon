import unittest

import numpy as np

from machaon.linalg import (
    ablate_input,
    ablate_output,
    bake_direction,
    clamp_rel_norm,
    low_rank_scale,
    nullspace_basis,
    nullspace_projector,
    project_complement,
    rank1,
    rank1_constrained,
    relative_norm,
)


def fixture(seed=0, d_in=24, d_out=16, n=12):
    rng = np.random.default_rng(seed)
    return (
        rng.normal(0, 0.02, size=(d_out, d_in)),
        rng.normal(0, 1, size=(n, d_in)),
        rng.normal(0, 1, size=(d_in,)),
        rng.normal(0, 1, size=(d_out,)),
    )


class TestRank1(unittest.TestCase):
    def test_maps_key_to_value_exactly(self):
        W, _, k, v = fixture()
        W2, delta = rank1(W, k, v)
        np.testing.assert_allclose(W2 @ k, v, atol=1e-10)
        np.testing.assert_allclose(W2, W + delta, atol=0)

    def test_does_not_preserve_protected_keys(self):
        W, P, k, v = fixture()
        W2, _ = rank1(W, k, v)
        drift = max(np.linalg.norm(W2 @ p - W @ p) for p in P)
        self.assertGreater(drift, 1e-6, "naive rank-1 should disturb protected keys")


class TestConstrained(unittest.TestCase):
    def test_maps_key_and_preserves_keys(self):
        W, P, k, v = fixture()
        W2, delta = rank1_constrained(W, k, v, preserve=P)
        np.testing.assert_allclose(W2 @ k, v, atol=1e-10)
        for p in P:
            self.assertLess(np.linalg.norm(delta @ p), 1e-10)

    def test_refuses_degenerate_target(self):
        W, P, k, v = fixture()
        with self.assertRaises(ValueError):
            rank1_constrained(W, P[0], v, preserve=P)

    def test_clamps_budget(self):
        W, P, k, v = fixture()
        W2, delta = rank1_constrained(W, k, v, preserve=P, max_rel_norm=0.01)
        self.assertLessEqual(relative_norm(delta, W), 0.01 + 1e-12)
        self.assertLessEqual(relative_norm(W2 - W, W), 0.01 + 1e-12)


class TestProjector(unittest.TestCase):
    def test_is_orthogonal_and_kills_span(self):
        _, P, _, _ = fixture()
        M = nullspace_projector(P)
        # Apple Accelerate raises spurious FP flags on plain matmul; the
        # numbers are exact, so only the flags are suppressed here.
        with np.errstate(all="ignore"):
            np.testing.assert_allclose(M @ M, M, atol=1e-10)  # idempotent
        np.testing.assert_allclose(M.T, M, atol=1e-12)        # symmetric
        for p in P:
            self.assertLess(np.linalg.norm(M @ p), 1e-10)

    def test_empty_preserve_is_identity(self):
        np.testing.assert_allclose(nullspace_projector(np.zeros((0, 7))), np.eye(7))

    def test_basis_matches_projector(self):
        _, P, k, _ = fixture()
        Q = nullspace_basis(P)
        self.assertEqual(Q.shape, (P.shape[1], P.shape[0]))
        with np.errstate(all="ignore"):  # Accelerate FP flags, see above
            np.testing.assert_allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=1e-12)
        np.testing.assert_allclose(
            project_complement(k, Q),
            nullspace_projector(P) @ k,
            atol=1e-10,
        )

    def test_basis_of_empty_set_is_empty(self):
        self.assertEqual(nullspace_basis(np.zeros((0, 7))).shape, (7, 0))


class TestAblation(unittest.TestCase):
    def test_output_direction_removed(self):
        W, _, _, _ = fixture()
        d = np.random.default_rng(1).normal(size=W.shape[0])
        W2, _ = ablate_output(W, d)
        dn = d / np.linalg.norm(d)
        self.assertLess(np.linalg.norm(dn @ W2), 1e-10)

    def test_input_direction_removed(self):
        W, _, _, _ = fixture()
        d = np.random.default_rng(2).normal(size=W.shape[1])
        W2, _ = ablate_input(W, d)
        dn = d / np.linalg.norm(d)
        self.assertLess(np.linalg.norm(W2 @ dn), 1e-10)


class TestBakeDirection(unittest.TestCase):
    def test_shifts_output_along_direction(self):
        W, _, _, _ = fixture()
        a = np.random.default_rng(3).normal(size=W.shape[1])
        d = np.random.default_rng(4).normal(size=W.shape[0])
        W2, _ = bake_direction(W, a, d, alpha=0.75)
        dn = d / np.linalg.norm(d)
        np.testing.assert_allclose(W2 @ a, W @ a + 0.75 * dn, atol=1e-10)


class TestClampAndRank(unittest.TestCase):
    def test_clamp_never_exceeds_budget(self):
        W, _, _, _ = fixture()
        delta = np.ones_like(W)
        out, rel = clamp_rel_norm(delta, W, 0.03)
        self.assertLessEqual(rel, 0.03 + 1e-12)
        self.assertLessEqual(relative_norm(out, W), 0.03 + 1e-12)

    def test_clamp_leaves_small_deltas_alone(self):
        W, _, _, _ = fixture()
        delta = np.zeros_like(W)
        out, _ = clamp_rel_norm(delta, W, 0.03)
        np.testing.assert_array_equal(out, delta)

    def test_low_rank_scale_truncates(self):
        rng = np.random.default_rng(5)
        delta = rng.normal(size=(10, 8))
        out = low_rank_scale(delta, rank=2)
        self.assertEqual(out.shape, delta.shape)
        self.assertLessEqual(np.linalg.matrix_rank(out), 2)

    def test_low_rank_scale_scales(self):
        rng = np.random.default_rng(6)
        u = rng.normal(size=(10, 1))
        v = rng.normal(size=(1, 8))
        delta = u @ v
        np.testing.assert_allclose(low_rank_scale(delta, 1, 3.0), 3.0 * delta, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
