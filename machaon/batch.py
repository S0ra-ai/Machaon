"""Batch least-squares weight edits with a null-space constraint.

MEMIT-style batch editing asks one question: can many key/value pairs share a
single delta and a single budget?  This module provides the smallest honest
version of that primitive:

    (W + delta) K^T = V^T

with an optional protected key set ``P`` and a relative-norm cap.  It is a
least-squares batch solver, not a reimplementation of MEMIT's covariance
hierarchy; every returned delta carries its exactness and clamping state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from ._fpe import quiet_blas
from .linalg import nullspace_basis, relative_norm

__all__ = ["BatchSolution", "solve_batch"]


@dataclass(frozen=True)
class BatchSolution:
    delta: np.ndarray
    relative_norm: float
    target_error: float
    per_key_error: tuple
    exact: bool
    clamped: bool
    rank: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "delta_shape": list(self.delta.shape),
            "relative_norm": self.relative_norm,
            "target_error": self.target_error,
            "per_key_error": list(self.per_key_error),
            "exact": self.exact,
            "clamped": self.clamped,
            "rank": self.rank,
        }


def _as_matrix(name: str, value: Any, width: Optional[int] = None):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("%s must be 2-D" % name)
    if not np.isfinite(array).all():
        raise ValueError("%s contains non-finite values" % name)
    if width is not None and array.shape[1] != width:
        raise ValueError("%s width %d does not match W width %d"
                         % (name, array.shape[1], width))
    return array


@quiet_blas
def solve_batch(
    W: Any,
    keys: Any,
    values: Any,
    *,
    preserve: Any = None,
    max_rel_norm: Optional[float] = None,
    clamp: bool = True,
    ridge: float = 0.0,
) -> BatchSolution:
    """Solve many key/value pairs with one minimum-norm delta.

    ``keys`` is ``(n, d_in)`` and ``values`` is ``(n, d_out)``.  The minimum
    Frobenius-norm solution is ``delta = (V^T - W K^T) (K K^T)^+ K``.  When
    ``preserve`` is supplied, the target keys are projected into the null
    space of the protected set, which makes ``delta @ P^T == 0`` exactly but
    can leave an unattainable residual on keys that overlap the protected
    subspace.

    If ``max_rel_norm`` is exceeded, ``clamp=True`` scales the delta down and
    reports ``exact=False``; ``clamp=False`` refuses it.  This is the same
    explicit trade-off as ``rank1_constrained``, made visible in the result.
    """
    W = _as_matrix("W", W)
    keys = _as_matrix("keys", keys, W.shape[1])
    values = _as_matrix("values", values)
    if values.shape[1] != W.shape[0]:
        raise ValueError("values width %d does not match W height %d"
                         % (values.shape[1], W.shape[0]))
    if keys.shape[0] != values.shape[0]:
        raise ValueError("keys and values must have the same row count")
    if keys.shape[0] == 0:
        raise ValueError("keys must contain at least one edit")
    if ridge < 0 or not np.isfinite(ridge):
        raise ValueError("ridge must be finite and non-negative")

    target_keys = keys
    if preserve is not None:
        protected = _as_matrix("preserve", preserve, W.shape[1])
        if protected.shape[0] == 0:
            protected = None
        else:
            Q = nullspace_basis(protected)
            target_keys = keys - (keys @ Q) @ Q.T

    residual = values.T - W @ keys.T
    gram = target_keys @ target_keys.T
    if ridge:
        gram = gram + float(ridge) * np.eye(gram.shape[0])
    # pinv handles collinear keys and the rank-deficient protected projection.
    delta = residual @ np.linalg.pinv(gram) @ target_keys
    rel = relative_norm(delta, W)
    clamped = False
    if max_rel_norm is not None:
        if not np.isfinite(max_rel_norm) or max_rel_norm <= 0:
            raise ValueError("max_rel_norm must be positive")
        if rel > max_rel_norm:
            if not clamp:
                raise ValueError(
                    "batch delta needs %.6f > budget %.6f"
                    % (rel, max_rel_norm)
                )
            delta = delta * (float(max_rel_norm) / rel)
            rel = float(max_rel_norm)
            clamped = True

    after = W + delta
    errors = np.linalg.norm(after @ keys.T - values.T, axis=0) / np.maximum(
        np.linalg.norm(values.T, axis=0), 1.0
    )
    target_error = float(np.linalg.norm(after @ keys.T - values.T)
                         / max(float(np.linalg.norm(values.T)), 1.0))
    exact = bool(target_error < 1e-10 and not clamped)
    return BatchSolution(
        delta=delta,
        relative_norm=rel,
        target_error=target_error,
        per_key_error=tuple(float(x) for x in errors),
        exact=exact,
        clamped=clamped,
        rank=int(np.linalg.matrix_rank(delta, tol=1e-10)),
    )
