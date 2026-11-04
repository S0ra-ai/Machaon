import unittest

from machaon import memory


class TestWeightsBytes(unittest.TestCase):
    def test_dtype_sizes(self):
        summary = {"dtypes": {"BF16": 1000, "F32": 10}}
        self.assertEqual(memory.weights_bytes(summary), 1000 * 2 + 10 * 4)

    def test_unknown_dtype_is_rejected(self):
        with self.assertRaises(ValueError):
            memory.weights_bytes({"dtypes": {"WEIRD": 1}})


class TestCheckFit(unittest.TestCase):
    def setUp(self):
        self._original = memory.available_bytes

    def tearDown(self):
        memory.available_bytes = self._original

    def test_refuses_when_over_budget(self):
        memory.available_bytes = lambda: 10_000
        report = memory.check_fit({"dtypes": {"BF16": 4_000}}, fraction=0.6)
        self.assertFalse(report["ok"])
        self.assertIn("exceed", report["reason"])

    def test_accepts_when_comfortably_under(self):
        memory.available_bytes = lambda: 10_000
        report = memory.check_fit({"dtypes": {"BF16": 2_000}}, fraction=0.6)
        self.assertTrue(report["ok"])

    def test_unknown_memory_is_not_a_refusal(self):
        memory.available_bytes = lambda: None
        report = memory.check_fit({"dtypes": {"BF16": 4_000}})
        self.assertTrue(report["ok"])
        self.assertTrue(report["unknown"])


class TestPlatform(unittest.TestCase):
    def test_available_bytes_is_positive_or_unknown(self):
        value = memory.available_bytes()
        self.assertTrue(value is None or value > 0)


if __name__ == "__main__":
    unittest.main()
