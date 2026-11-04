"""The file-level operations behind the MCP server."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from machaon import tools

try:
    import torch
    from safetensors.torch import save_file
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

FIXTURE = Path(__file__).parent / "fixtures" / "openwsp_v1"
requires_torch = unittest.skipUnless(HAVE_TORCH, "torch + safetensors not installed")


def _originals():
    root = FIXTURE / "originals"
    return {p.name[: -len(".npy")]: np.load(p) for p in sorted(root.glob("*.npy"))}


@requires_torch
class TestTools(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.base = self.dir / "model.safetensors"
        save_file({k: torch.from_numpy(v) for k, v in _originals().items()}, str(self.base))

    def tearDown(self):
        self.tmp.cleanup()

    def test_describe_checkpoint(self):
        summary = tools.describe_checkpoint(str(self.base))
        self.assertEqual(summary["tensors"], 2)
        self.assertEqual(summary["shards"], 1)
        self.assertGreater(summary["parameters"], 0)

    def test_list_tensors_filters(self):
        rows = tools.list_tensors(str(self.base), suffix="down_proj.weight")
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(tools.list_tensors(str(self.base), layer=0)), 1)
        self.assertEqual(len(tools.list_tensors(str(self.base), contains="nothing")), 0)

    def test_list_tensors_respects_limit(self):
        self.assertEqual(len(tools.list_tensors(str(self.base), limit=1)), 1)

    def test_audit_reference_edit(self):
        report = tools.audit_checkpoint_edit(str(self.base), str(FIXTURE))
        self.assertTrue(report["passed"])
        self.assertEqual(report["changed_tensors"], 2)
        self.assertGreater(report["worst_relative_change"], 0.0)
        golden = json.loads((FIXTURE / "manifest.json").read_text())
        self.assertEqual(report["edit_sha256"], golden["edit_sha256"])

    def test_audit_can_refuse_on_budget(self):
        report = tools.audit_checkpoint_edit(str(self.base), str(FIXTURE), max_rel_norm=1e-9)
        self.assertFalse(report["passed"])

    def test_apply_delta_roundtrip(self):
        name = "model.layers.0.mlp.down_proj.weight"
        base = tools.list_tensors(str(self.base), contains=name)[0]
        delta = np.zeros(base["shape"], dtype=np.float32)
        delta[0, 0] = 0.01
        delta_path = self.dir / "delta.npy"
        np.save(delta_path, delta)

        out = self.dir / "edited"
        report = tools.apply_delta(str(self.base), name, str(delta_path), str(out))
        self.assertEqual(report["tensor"], name)
        self.assertGreater(report["relative_change"], 0.0)
        self.assertTrue((out / "model.safetensors").exists())

    def test_gate_metrics(self):
        good = {"rewrite_acc": 1.0, "paraphrase_acc": 0.9, "unrelated_acc": 1.0,
                "kl_p95": 0.01, "kl_max": 0.1, "mmlu_delta": 0.0,
                "ppl_ratio": 1.0, "longform_contradictions": 0}
        self.assertTrue(tools.gate_metrics(good)["passed"])
        self.assertFalse(tools.gate_metrics({})["passed"])
        self.assertIn("rewrite_acc", tools.gate_metrics({})["missing"])


if __name__ == "__main__":
    unittest.main()
