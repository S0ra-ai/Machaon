"""Deterministic weight fingerprints and edit-diff reports.

These helpers answer a narrower question than a signature: did these exact
arrays change, where, and by how much? They are useful for an audit log or a
pre/post checkpoint comparison. They do not establish authenticity; use an
external signature over the fingerprint if that is required.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping

import numpy as np

__all__ = ["TensorFingerprint", "tensor_fingerprint", "model_fingerprint", "diff_tensors"]


def _array(value: Any, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("%s contains non-finite values" % name)
    return array


def _digest(array: np.ndarray) -> str:
    header = json.dumps(
        {"shape": list(array.shape), "dtype": "float64"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    body = np.ascontiguousarray(array, dtype="<f8").tobytes()
    return hashlib.sha256(header + b"\0" + body).hexdigest()


@dataclass(frozen=True)
class TensorFingerprint:
    sha256: str
    shape: tuple
    size: int
    norm: float
    mean: float
    std: float
    zero_fraction: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sha256": self.sha256,
            "shape": list(self.shape),
            "size": self.size,
            "norm": self.norm,
            "mean": self.mean,
            "std": self.std,
            "zero_fraction": self.zero_fraction,
        }


def tensor_fingerprint(value: Any) -> TensorFingerprint:
    """Stable float64 content address plus descriptive statistics."""
    array = _array(value, "tensor")
    return TensorFingerprint(
        sha256=_digest(array),
        shape=tuple(array.shape),
        size=int(array.size),
        norm=float(np.linalg.norm(array)),
        mean=float(np.mean(array)) if array.size else 0.0,
        std=float(np.std(array)) if array.size else 0.0,
        zero_fraction=float(np.mean(array == 0.0)) if array.size else 0.0,
    )


def model_fingerprint(weights: Mapping[str, Any]) -> Dict[str, Any]:
    """Content address a named tensor collection without loading a model."""
    if not isinstance(weights, Mapping) or not weights:
        raise ValueError("weights must be a non-empty mapping")
    tensors = {}
    records = []
    for name in sorted(weights):
        fingerprint = tensor_fingerprint(weights[name])
        tensors[str(name)] = fingerprint.to_dict()
        records.append({
            "name": str(name),
            "sha256": fingerprint.sha256,
            "shape": list(fingerprint.shape),
        })
    digest = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "format": "machaon.weight-fingerprint",
        "format_version": 1,
        "sha256": digest,
        "n_tensors": len(tensors),
        "parameters": int(sum(item["size"] for item in tensors.values())),
        "tensors": tensors,
    }


def diff_tensors(base: Any, changed: Any) -> Dict[str, Any]:
    """Describe exactly where two tensors differ."""
    before = _array(base, "base")
    after = _array(changed, "changed")
    if before.shape != after.shape:
        raise ValueError("base and changed shapes differ")
    delta = after - before
    indices = np.flatnonzero(delta.reshape(-1) != 0.0)
    base_norm = float(np.linalg.norm(before))
    delta_norm = float(np.linalg.norm(delta))
    report: Dict[str, Any] = {
        "changed": bool(indices.size),
        "changed_entries": int(indices.size),
        "support_fraction": (
            float(indices.size) / int(before.size) if before.size else 0.0
        ),
        "relative_change": delta_norm / base_norm if base_norm > 0.0 else None,
        "max_abs_change": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "base": tensor_fingerprint(before).to_dict(),
        "changed_fingerprint": tensor_fingerprint(after).to_dict(),
    }
    if before.ndim == 2 and indices.size:
        rows, columns = np.unravel_index(indices, before.shape)
        report["changed_rows"] = int(len(np.unique(rows)))
        report["changed_columns"] = int(len(np.unique(columns)))
    return report
