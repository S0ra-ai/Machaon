import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from machaon.adapters import Checkpoint
from machaon.linalg import rank1_constrained
from machaon.verify import check_protected

try:
    import torch
    from safetensors.torch import save_file
    HAVE_TORCH = True
except ImportError:  # the numpy-only CI job runs everything else
    torch = None
    save_file = None
    HAVE_TORCH = False

requires_torch = unittest.skipUnless(HAVE_TORCH, "torch + safetensors not installed")


def _write(path, tensors, metadata=None):
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path), metadata=metadata)


@requires_torch
class TestSingleFile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.rng = np.random.default_rng(0)
        self.w = self.rng.normal(0, 0.02, size=(16, 12)).astype(np.float32)
        self.other = self.rng.normal(0, 0.02, size=(5, 3)).astype(np.float32)
        self.path = self.dir / "model.safetensors"
        _write(self.path, {
            "model.layers.0.mlp.down_proj.weight": torch.from_numpy(self.w),
            "model.layers.1.mlp.down_proj.weight": torch.from_numpy(self.other),
        }, metadata={"format": "pt"})

    def tearDown(self):
        self.tmp.cleanup()

    def test_names_and_info(self):
        ckpt = Checkpoint(self.path)
        self.assertEqual(len(ckpt.names()), 2)
        info = ckpt.info("model.layers.0.mlp.down_proj.weight")
        self.assertEqual(info.shape, (16, 12))
        self.assertEqual(info.dtype, "F32")

    def test_read_roundtrip(self):
        ckpt = Checkpoint(self.path)
        np.testing.assert_array_equal(
            ckpt.read("model.layers.0.mlp.down_proj.weight"), self.w)

    def test_find(self):
        ckpt = Checkpoint(self.path)
        self.assertEqual(len(ckpt.find(layer=0)), 1)
        self.assertEqual(len(ckpt.find(suffix="down_proj.weight")), 2)
        self.assertEqual(len(ckpt.find(contains="nothing")), 0)

    def test_write_preserves_untouched_tensors(self):
        ckpt = Checkpoint(self.path)
        delta = np.zeros_like(self.w)
        delta[0, 0] = 0.01
        out = self.dir / "edited"
        report = ckpt.write_with({"model.layers.0.mlp.down_proj.weight": delta}, out)

        self.assertEqual(len(report["files"]), 1)
        edited = Checkpoint(out / "model.safetensors")
        np.testing.assert_allclose(
            edited.read("model.layers.0.mlp.down_proj.weight"), self.w + delta, atol=1e-7)
        np.testing.assert_array_equal(
            edited.read("model.layers.1.mlp.down_proj.weight"), self.other)

    def test_unknown_delta_rejected(self):
        ckpt = Checkpoint(self.path)
        with self.assertRaises(KeyError):
            ckpt.write_with({"nope": np.zeros((2, 2))}, self.dir / "x")

    def test_shape_mismatch_rejected(self):
        ckpt = Checkpoint(self.path)
        with self.assertRaises(ValueError):
            ckpt.write_with(
                {"model.layers.0.mlp.down_proj.weight": np.zeros((2, 2))},
                self.dir / "x")

    def test_bfloat16_roundtrip(self):
        path = self.dir / "bf16.safetensors"
        bf = torch.randn(8, 6, dtype=torch.bfloat16)
        _write(path, {"w": bf})
        ckpt = Checkpoint(path)
        read = ckpt.read("w")
        self.assertEqual(read.dtype, np.float32)
        np.testing.assert_allclose(read, bf.float().numpy(), atol=0)

        delta = np.zeros((8, 6), dtype=np.float64)
        delta[1, 2] = 0.05
        out = self.dir / "bf16_out"
        ckpt.write_with({"w": delta}, out)
        with __import__("safetensors").safe_open(str(out / "bf16.safetensors"), framework="pt") as f:
            self.assertEqual(str(f.get_slice("w").get_dtype()), "BF16")


@requires_torch
class TestSharded(unittest.TestCase):
    def test_sharded_read_and_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(1)
            a = rng.normal(size=(4, 3)).astype(np.float32)
            b = rng.normal(size=(3, 2)).astype(np.float32)
            _write(root / "model-00001.safetensors", {"a": torch.from_numpy(a)})
            _write(root / "model-00002.safetensors", {"b": torch.from_numpy(b)})
            (root / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {"a": "model-00001.safetensors",
                               "b": "model-00002.safetensors"}}))

            ckpt = Checkpoint(root)
            self.assertEqual(ckpt.names(), ["a", "b"])
            np.testing.assert_array_equal(ckpt.read("b"), b)

            out = root / "edited"
            ckpt.write_with({"b": np.full((3, 2), 0.5)}, out)
            edited = Checkpoint(out)
            np.testing.assert_allclose(edited.read("b"), b + 0.5, atol=1e-7)
            np.testing.assert_array_equal(edited.read("a"), a)


@requires_torch
class TestEndToEnd(unittest.TestCase):
    """linalg -> checkpoint -> disk -> reload, with the protected set checked."""

    def test_rank1_edit_survives_a_disk_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(2)
            W = rng.normal(0, 0.5, size=(32, 24))
            keys = rng.normal(size=(8, 24))
            k = rng.normal(size=24)
            v = W @ k + rng.normal(0, 1.5, size=32)

            path = root / "m.safetensors"
            _write(path, {"model.layers.3.mlp.down_proj.weight": torch.from_numpy(
                W.astype(np.float32))})

            ckpt = Checkpoint(path)
            base = ckpt.read("model.layers.3.mlp.down_proj.weight", dtype=np.float64)
            _, delta = rank1_constrained(base, k, v, preserve=keys, max_rel_norm=0.05)
            ckpt.write_with({"model.layers.3.mlp.down_proj.weight": delta}, root / "out")

            reloaded = Checkpoint(root / "out").read(
                "model.layers.3.mlp.down_proj.weight", dtype=np.float64)
            applied = reloaded - base
            report = check_protected(base, applied, keys, max_drift=1e-5)
            self.assertTrue(report["passed"], report)
            self.assertLess(np.linalg.norm(applied @ k - delta @ k) / np.linalg.norm(delta @ k), 1e-5)


if __name__ == "__main__":
    unittest.main()
