import unittest

import numpy as np

from machaon.refusals import EditRefused, check_tensor


class TestRefusals(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.base = self.rng.normal(0, 0.02, size=(5, 4)).astype(np.float32)

    def test_legitimate_sparse_edit_accepted(self):
        value = self.base.copy()
        value[2, 3] += 0.01
        out = check_tensor("w", value, self.base)
        np.testing.assert_array_equal(out, value)

    def test_frozen_tensor_refused(self):
        with self.assertRaises(EditRefused):
            check_tensor("lm_head.weight", self.base, self.base, frozen={"lm_head.weight"})

    def test_shape_mismatch_refused(self):
        with self.assertRaises(EditRefused):
            check_tensor("w", np.zeros((2, 2)), self.base)

    def test_non_finite_refused(self):
        value = self.base.copy()
        value[0, 0] = np.nan
        with self.assertRaises(EditRefused):
            check_tensor("w", value, self.base)

    def test_bf16_overflow_refused(self):
        value = self.base.copy()
        with np.errstate(over="ignore"):
            value[0, 0] = 1e39
        with self.assertRaises(EditRefused):
            check_tensor("w", value, self.base)

    def test_whole_tensor_rescale_refused(self):
        with self.assertRaises(EditRefused):
            check_tensor("w", self.base * np.float32(1.5), self.base)

    def test_rescale_allowed_when_disabled(self):
        out = check_tensor("w", self.base * np.float32(1.5), self.base, refuse_rescale=False)
        self.assertEqual(out.shape, self.base.shape)

    def test_identity_accepted(self):
        out = check_tensor("w", self.base, self.base)
        np.testing.assert_array_equal(out, self.base)


if __name__ == "__main__":
    unittest.main()
