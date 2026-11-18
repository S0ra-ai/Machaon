"""Small task-vector primitives.

Task arithmetic treats ``tau = theta_finetuned - theta_base`` as a reusable
direction.  This module keeps the operation explicit: shape validation,
positive/negative coefficients, a relative-norm budget and a reported clamp.
It does not implement mergekit's merge recipes, model architecture alignment or
held-out evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np

from ._fpe import quiet_blas

__all__ = ["TaskVector", "TaskVectorCombination", "task_vector", "combine_task_vectors"]


def _array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("%s must be 2-D" % name)
    if not np.isfinite(array).all():
        raise ValueError("%s contains non-finite values" % name)
    return array


@dataclass(frozen=True)
class TaskVector:
    delta: np.ndarray
    relative_norm: float
    base_norm: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shape": list(self.delta.shape),
            "relative_norm": float(self.relative_norm),
            "base_norm": float(self.base_norm),
        }


@dataclass(frozen=True)
class TaskVectorCombination:
    delta: np.ndarray
    relative_norm: Optional[float]
    terms: Mapping[str, float]
    clamped: bool
    base_norm: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shape": list(self.delta.shape),
            "relative_norm": self.relative_norm,
            "terms": {str(k): float(v) for k, v in self.terms.items()},
            "clamped": bool(self.clamped),
            "base_norm": self.base_norm,
        }


@quiet_blas
def task_vector(base: Any, finetuned: Any) -> TaskVector:
    """Return ``finetuned - base`` with its relative magnitude."""
    base = _array(base, "base")
    finetuned = _array(finetuned, "finetuned")
    if base.shape != finetuned.shape:
        raise ValueError("base and finetuned must have the same shape")
    delta = finetuned - base
    base_norm = float(np.linalg.norm(base))
    if base_norm == 0.0:
        raise ValueError("base has zero norm")
    return TaskVector(
        delta=delta,
        relative_norm=float(np.linalg.norm(delta) / base_norm),
        base_norm=base_norm,
    )


def _as_delta(value: Any, name: str, shape: tuple) -> np.ndarray:
    delta = value.delta if isinstance(value, TaskVector) else value
    delta = _array(delta, name)
    if delta.shape != shape:
        raise ValueError("%s shape %r does not match %r" % (name, delta.shape, shape))
    return delta


@quiet_blas
def combine_task_vectors(
    vectors: Mapping[str, Any],
    coefficients: Optional[Mapping[str, float]] = None,
    *,
    base: Any = None,
    max_rel_norm: Optional[float] = None,
    clamp: bool = True,
) -> TaskVectorCombination:
    """Combine task vectors by coefficient, optionally enforcing a budget.

    With ``base`` supplied, ``relative_norm`` is measured against that base.
    If the budget is exceeded, ``clamp=True`` scales the combined delta and
    reports it; ``clamp=False`` raises.  Without a base, budget checks are not
    possible and ``relative_norm`` is ``None``.
    """
    if not isinstance(vectors, Mapping) or not vectors:
        raise ValueError("vectors must be a non-empty mapping")
    coefficients = dict(coefficients or {})
    first_name = sorted(vectors)[0]
    first = vectors[first_name]
    shape = (
        tuple(first.delta.shape) if isinstance(first, TaskVector)
        else tuple(np.asarray(first).shape)
    )
    terms = {}
    combined = np.zeros(shape, dtype=np.float64)
    for name in sorted(vectors):
        coefficient = float(coefficients.get(name, 1.0))
        if not np.isfinite(coefficient):
            raise ValueError("coefficient for %r must be finite" % name)
        delta = _as_delta(vectors[name], name, shape)
        combined += coefficient * delta
        terms[name] = coefficient

    base_array = None
    rel = None
    base_norm = None
    if base is not None:
        base_array = _array(base, "base")
        if base_array.shape != shape:
            raise ValueError("base shape %r does not match %r" % (base_array.shape, shape))
        base_norm = float(np.linalg.norm(base_array))
        if base_norm == 0.0:
            raise ValueError("base has zero norm")
        rel = float(np.linalg.norm(combined) / base_norm)

    clamped = False
    if max_rel_norm is not None:
        if base_array is None:
            raise ValueError("max_rel_norm requires a base tensor")
        if not np.isfinite(float(max_rel_norm)) or float(max_rel_norm) <= 0.0:
            raise ValueError("max_rel_norm must be positive")
        if rel > float(max_rel_norm):
            if not clamp:
                raise ValueError(
                    "combined task vector needs %.6f > budget %.6f"
                    % (rel, max_rel_norm)
                )
            combined *= float(max_rel_norm) / rel
            rel = float(max_rel_norm)
            clamped = True

    return TaskVectorCombination(
        delta=combined,
        relative_norm=rel,
        terms=terms,
        clamped=clamped,
        base_norm=base_norm,
    )
