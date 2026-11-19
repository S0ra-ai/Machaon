"""The device probe never imports or allocates a model."""
from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

from machaon.devices import device_report


class TestDeviceReport(unittest.TestCase):
    def test_shape_without_torch(self):
        with mock.patch.dict(sys.modules, {"torch": None}):
            report = device_report()
        self.assertEqual(report["default"], "cpu")
        self.assertFalse(report["torch"])
        self.assertFalse(report["cuda"])
        self.assertFalse(report["mps"])

    def test_reports_a_visible_cuda_and_mps(self):
        torch_module = types.ModuleType("torch")
        torch_module.cuda = types.SimpleNamespace(
            is_available=lambda: True, device_count=lambda: 2
        )
        torch_module.backends = types.SimpleNamespace(
            mps=types.SimpleNamespace(
                is_available=lambda: True, is_built=lambda: True
            )
        )
        with mock.patch.dict(sys.modules, {"torch": torch_module}):
            report = device_report()
        self.assertTrue(report["torch"])
        self.assertTrue(report["cuda"])
        self.assertEqual(report["cuda_count"], 2)
        self.assertTrue(report["mps"])
        self.assertTrue(report["mps_built"])

    def test_a_broken_backend_does_not_raise(self):
        def boom():
            raise RuntimeError("no driver")

        torch_module = types.ModuleType("torch")
        torch_module.cuda = types.SimpleNamespace(
            is_available=boom, device_count=boom
        )
        torch_module.backends = types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=boom, is_built=boom)
        )
        with mock.patch.dict(sys.modules, {"torch": torch_module}):
            report = device_report()
        self.assertFalse(report["cuda"])
        self.assertFalse(report["mps"])


if __name__ == "__main__":
    unittest.main()
