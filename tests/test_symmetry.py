"""Permutation and positive-scaling alignment for weight tensors."""
import unittest

import numpy as np

from machaon.symmetry import (
    align_mlp_hidden,
    align_rows,
    functional_alignment_check,
    merge_mlp_aligned,
    mlp_outputs,
    permute_columns,
    permute_rows,
)


class TestPermutations(unittest.TestCase):
    def test_roundtrip_rows_and_columns(self):
        rng = np.random.default_rng(1)
        W = rng.normal(size=(6, 4))
        rows = [3, 0, 5, 1, 4, 2]
        cols = [2, 0, 3, 1]
        np.testing.assert_array_equal(
            permute_rows(permute_rows(W, [0, 1, 2, 3, 4, 5]), rows),
            W[rows],
        )
        np.testing.assert_array_equal(
            permute_columns(W, cols), W[:, cols]
        )

    def test_bad_permutations_are_refused(self):
        with self.assertRaises(ValueError):
            permute_rows(np.zeros((3, 2)), [0, 0, 1])
        with self.assertRaises(ValueError):
            permute_columns(np.zeros((3, 2)), [0])


class TestAlignment(unittest.TestCase):
    def test_rows_recover_a_permutation(self):
        rng = np.random.default_rng(3)
        ref = rng.normal(size=(5, 4))
        order = np.array([2, 4, 0, 3, 1])
        cand = ref[order]
        result = align_rows(ref, cand)
        np.testing.assert_array_equal(result.aligned, ref)
        self.assertTrue(result.passed)
        self.assertAlmostEqual(result.min_similarity, 1.0, places=12)
        self.assertFalse(result.metadata["optimal"])

    def test_columns_recover_a_permutation(self):
        rng = np.random.default_rng(4)
        ref = rng.normal(size=(3, 5))
        cand = ref[:, [4, 1, 3, 0, 2]]
        result = align_rows(ref, cand, axis=1)
        np.testing.assert_array_equal(result.aligned, ref)
        self.assertEqual(result.metadata["axis"], 1)

    def test_positive_scaling_is_cosine_aligned(self):
        rng = np.random.default_rng(5)
        ref = rng.normal(size=(4, 3))
        cand = ref.copy()
        cand[1] *= 3.0
        cand[3] *= 0.25
        result = align_rows(ref, cand, min_similarity=0.999)
        self.assertTrue(result.passed)
        self.assertTrue(np.allclose(result.similarities, 1.0))

    def test_zero_units_are_reported(self):
        ref = np.array([[1.0, 0.0], [0.0, 1.0]])
        cand = np.array([[0.0, 0.0], [0.0, 1.0]])
        result = align_rows(ref, cand)
        self.assertEqual(result.metadata["zero_units"], 1)

    def test_shape_and_unit_limits_are_refused(self):
        with self.assertRaises(ValueError):
            align_rows(np.zeros((2, 3)), np.zeros((3, 2)))
        with self.assertRaises(ValueError):
            align_rows(np.zeros((3, 2)), np.zeros((3, 2)), max_units=2)


class TestMLPAlignment(unittest.TestCase):
    def test_hidden_permutation_is_applied_to_both_matrices(self):
        rng = np.random.default_rng(7)
        ref_in = rng.normal(size=(5, 4))
        ref_out = rng.normal(size=(3, 5))
        order = np.array([3, 0, 4, 1, 2])
        cand_in = ref_in[order]
        cand_out = ref_out[:, order]
        result = align_mlp_hidden(ref_in, ref_out, cand_in, cand_out)
        np.testing.assert_array_equal(result.incoming, ref_in)
        np.testing.assert_array_equal(result.outgoing, ref_out)
        payload = result.to_dict()
        self.assertEqual(payload["units"]["permutation"], [1, 3, 4, 0, 2])

    def test_inconsistent_shapes_are_refused(self):
        with self.assertRaises(ValueError):
            align_mlp_hidden(
                np.zeros((4, 3)), np.zeros((2, 4)),
                np.zeros((4, 3)), np.zeros((2, 5)),
            )

    def test_aligned_merge_recovers_both_endpoints(self):
        rng = np.random.default_rng(8)
        ref_in = rng.normal(size=(4, 3))
        ref_out = rng.normal(size=(2, 4))
        order = np.array([2, 0, 3, 1])
        cand_in = ref_in[order]
        cand_out = ref_out[:, order]
        only_ref = merge_mlp_aligned(
            ref_in, ref_out, cand_in, cand_out, alpha=0.0
        )
        np.testing.assert_array_equal(only_ref.incoming, ref_in)
        np.testing.assert_array_equal(only_ref.outgoing, ref_out)
        only_cand = merge_mlp_aligned(
            ref_in, ref_out, cand_in, cand_out, alpha=1.0
        )
        np.testing.assert_array_equal(only_cand.incoming, ref_in)
        np.testing.assert_array_equal(only_cand.outgoing, ref_out)

    def test_aligned_merge_is_not_a_naive_coordinate_average(self):
        rng = np.random.default_rng(9)
        ref_in = rng.normal(size=(4, 3))
        ref_out = rng.normal(size=(2, 4))
        order = np.array([2, 0, 3, 1])
        cand_in = ref_in[order]
        cand_out = ref_out[:, order]
        merged = merge_mlp_aligned(
            ref_in, ref_out, cand_in, cand_out, alpha=0.5
        )
        np.testing.assert_allclose(merged.incoming, ref_in, atol=0)
        np.testing.assert_allclose(merged.outgoing, ref_out, atol=0)
        self.assertFalse(np.array_equal(
            (ref_in + cand_in) / 2.0, merged.incoming
        ))

    def test_aligned_merge_interpolates_after_permutation(self):
        rng = np.random.default_rng(12)
        ref_in = rng.normal(size=(4, 3))
        ref_out = rng.normal(size=(2, 4))
        candidate_in = ref_in + 0.1 * rng.normal(size=ref_in.shape)
        candidate_out = ref_out + 0.1 * rng.normal(size=ref_out.shape)
        order = np.array([2, 0, 3, 1])
        inverse = np.argsort(order)
        merged = merge_mlp_aligned(
            ref_in, ref_out,
            candidate_in[order], candidate_out[:, order],
            alpha=0.5,
        )
        np.testing.assert_allclose(
            merged.incoming, (ref_in + candidate_in) / 2.0, atol=1e-12
        )
        np.testing.assert_allclose(
            merged.outgoing, (ref_out + candidate_out) / 2.0, atol=1e-12
        )
        self.assertTrue(np.array_equal(merged.units.permutation, inverse))

    def test_merge_alpha_is_validated(self):
        with self.assertRaises(ValueError):
            merge_mlp_aligned(
                np.zeros((2, 2)), np.zeros((2, 2)),
                np.zeros((2, 2)), np.zeros((2, 2)), alpha=1.1,
            )

    def test_functional_check_passes_for_an_exact_permutation(self):
        rng = np.random.default_rng(10)
        ref_in = rng.normal(size=(5, 4))
        ref_out = rng.normal(size=(3, 5))
        order = np.array([3, 0, 4, 1, 2])
        probes = rng.normal(size=(12, 4))
        report = functional_alignment_check(
            ref_in, ref_out, ref_in[order], ref_out[:, order], probes
        )
        self.assertTrue(report["passed"])
        self.assertLess(report["max_abs_error"], 1e-10)
        self.assertEqual(report["n_probes"], 12)

    def test_functional_check_catches_a_different_network(self):
        rng = np.random.default_rng(11)
        ref_in = rng.normal(size=(4, 3))
        ref_out = rng.normal(size=(2, 4))
        probes = rng.normal(size=(8, 3))
        report = functional_alignment_check(
            ref_in, ref_out,
            rng.normal(size=ref_in.shape),
            rng.normal(size=ref_out.shape),
            probes,
        )
        self.assertFalse(report["passed"])
        self.assertGreater(report["max_abs_error"], 1e-9)

    def test_mlp_outputs_validates_shape_and_activation(self):
        with self.assertRaises(ValueError):
            mlp_outputs(np.zeros((4, 3)), np.zeros((2, 4)), np.zeros((1, 2)))
        with self.assertRaises(ValueError):
            mlp_outputs(
                np.zeros((4, 3)), np.zeros((2, 4)),
                np.zeros((1, 3)), activation="tanh",
            )


if __name__ == "__main__":
    unittest.main()
