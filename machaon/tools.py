"""Operations an agent can call without loading a model.

These are plain functions: no transport, no globals, no model state. The MCP
server and the CLI are thin wrappers around them, and the tests exercise them
directly.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .adapters import Checkpoint
from .hygieia.gates import evaluate
from .interop import openwsp
from .refusals import check_tensor
from .verify import audit_edit

__all__ = [
    "describe_checkpoint",
    "list_tensors",
    "audit_checkpoint_edit",
    "apply_delta",
    "gate_metrics",
]


def describe_checkpoint(path: str) -> Dict[str, Any]:
    """Tensor count, parameter count, dtype mix and shard count. Header only."""
    return Checkpoint(path).summary()


def list_tensors(path: str, contains: Optional[str] = None,
                 suffix: Optional[str] = None, layer: Optional[int] = None,
                 limit: int = 50) -> List[Dict[str, Any]]:
    """Bounded tensor listing; values are never returned."""
    checkpoint = Checkpoint(path)
    names = checkpoint.find(contains=contains, suffix=suffix, layer=layer)
    out = []
    for name in names[: max(1, int(limit))]:
        info = checkpoint.info(name)
        out.append({"name": name, "shape": list(info.shape), "dtype": info.dtype})
    return out


def audit_checkpoint_edit(base_path: str, edit_dir: str,
                          max_rel_norm: Optional[float] = None) -> Dict[str, Any]:
    """Verify an OpenWSP v1 edit against a base checkpoint, then report drift.

    This is the operation that does not require trusting the tool that produced
    the edit: hashes are checked against the base, and the relative change is
    recomputed independently.
    """
    manifest_path = Path(edit_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    names = [record["name"] for record in manifest.get("tensors", [])]
    if not names:
        return {"passed": True, "changed_tensors": 0, "tensors": {},
                "edit_sha256": manifest.get("edit_sha256")}

    checkpoint = Checkpoint(base_path)
    originals = {name: checkpoint.read(name) for name in names}
    verified, rebuilt = openwsp.read_edit(edit_dir, originals)
    deltas = openwsp.delta_from_edit(verified, rebuilt, originals)

    report = audit_edit(deltas, originals)
    worst = report["worst_relative_change"]
    within = max_rel_norm is None or worst <= max_rel_norm
    return {
        "passed": bool(within),
        "edit_sha256": verified.get("edit_sha256"),
        "changed_tensors": int(verified.get("changed_tensors", len(deltas))),
        "worst_relative_change": worst,
        "max_rel_norm": max_rel_norm,
        "tensors": report["tensors"],
    }


def apply_delta(base_path: str, tensor: str, delta_path: str, out_dir: str,
                refuse_rescale: bool = True) -> Dict[str, Any]:
    """Apply one ``.npy`` delta to one tensor and write a new checkpoint."""
    checkpoint = Checkpoint(base_path)
    base = checkpoint.read(tensor)
    delta = np.load(delta_path)
    if delta.shape != base.shape:
        raise ValueError("delta shape %r does not match %r" % (delta.shape, base.shape))

    checked = check_tensor(tensor, base + delta, base, refuse_rescale=refuse_rescale)
    report = checkpoint.write_with({tensor: checked - base}, out_dir)
    rel = float(np.linalg.norm(delta) / np.linalg.norm(base)) if np.linalg.norm(base) else None
    return {
        "tensor": tensor,
        "relative_change": rel,
        "delta_sha256": hashlib.sha256(np.ascontiguousarray(delta).tobytes()).hexdigest(),
        "output": str(Path(out_dir)),
        "files": report["files"],
    }


def gate_metrics(metrics: Dict[str, float]) -> Dict[str, Any]:
    """Apply the default acceptance gates. Missing metrics count as failures."""
    verdict = evaluate(metrics)
    return {
        "passed": verdict.passed,
        "failures": verdict.failures,
        "missing": verdict.missing,
        "n_gates": verdict.n_gates,
        "summary": verdict.summary(),
    }
