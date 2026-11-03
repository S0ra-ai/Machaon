"""Post-hoc verification of an edit against a protected key set.

These checks do not assume the edit was produced by Machaon.  Any delta — from
a rank-one solve, from another tool, or from a hand-written patch — can be
audited: *did this edit move the things it promised not to move?*
"""
from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np

from ._fpe import quiet_blas

__all__ = ["relative_change", "protected_drift", "check_protected", "audit_edit"]


def _arr(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


@quiet_blas
def relative_change(base: Any, delta: Any) -> float:
    """||delta||_F / ||base||_F."""
    base = _arr(base)
    delta = _arr(delta)
    if base.shape != delta.shape:
        raise ValueError("shape mismatch: %r != %r" % (base.shape, delta.shape))
    denom = float(np.linalg.norm(base))
    if denom == 0.0:
        raise ValueError("base tensor has zero norm")
    return float(np.linalg.norm(delta) / denom)


@quiet_blas
def protected_drift(base: Any, delta: Any, keys: Any, floor: float = 1e-12) -> np.ndarray:
    """Relative output change on each protected key.

    ``drift[i] = ||delta @ k_i|| / max(||base @ k_i||, floor)``

    For an edit built with Machaon's null-space constraint this is at machine
    precision (1e-16); for an unconstrained edit it is macroscopic.
    """
    base = _arr(base)
    delta = _arr(delta)
    keys = np.atleast_2d(_arr(keys))
    if base.shape != delta.shape:
        raise ValueError("base and delta must have the same shape")
    if keys.shape[1] != base.shape[1]:
        raise ValueError(
            "keys have width %d, tensor expects %d" % (keys.shape[1], base.shape[1])
        )
    before = np.linalg.norm(base @ keys.T, axis=0)
    moved = np.linalg.norm(delta @ keys.T, axis=0)
    return moved / np.maximum(before, floor)


def check_protected(
    base: Any, delta: Any, keys: Any, max_drift: float = 1e-6
) -> Dict[str, Any]:
    """Pass/fail verdict for the protected key set."""
    drift = protected_drift(base, delta, keys)
    failed = [int(i) for i in np.flatnonzero(drift > max_drift)]
    return {
        "passed": not failed,
        "max_drift": float(drift.max()) if drift.size else 0.0,
        "mean_drift": float(drift.mean()) if drift.size else 0.0,
        "p95_drift": float(np.quantile(drift, 0.95)) if drift.size else 0.0,
        "failed_keys": failed,
        "n_keys": int(drift.size),
        "threshold": float(max_drift),
    }


def audit_edit(
    deltas: Mapping[str, Any],
    bases: Mapping[str, Any],
    protected: Mapping[str, Any] | None = None,
    max_drift: float = 1e-6,
) -> Dict[str, Any]:
    """Summarise a whole edit: spend per tensor plus protected-key verdicts."""
    protected = protected or {}
    tensors: Dict[str, Any] = {}
    worst = 0.0
    all_passed = True
    for name, delta in deltas.items():
        if name not in bases:
            raise KeyError("no base tensor for %r" % name)
        entry = {"relative_change": relative_change(bases[name], delta)}
        worst = max(worst, entry["relative_change"])
        if name in protected:
            check = check_protected(bases[name], delta, protected[name], max_drift)
            entry["protected"] = check
            all_passed = all_passed and check["passed"]
        tensors[name] = entry
    return {
        "passed": all_passed,
        "worst_relative_change": worst,
        "tensors": tensors,
    }
