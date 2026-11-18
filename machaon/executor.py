"""Execute the declarative edit DSL.

`EditSpec` is the audit record.  This module is the small, explicit bridge from
that record to the pure NumPy primitives in :mod:`machaon.linalg`.

The executor never loads a model and never writes a checkpoint.  It takes a
weight matrix plus the operands the selected operation needs, returns the
delta, and reports the relative spend.  That keeps the DSL
usable from a notebook, a checkpoint script, or an agent tool without giving
the caller ambient access to training or model state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from .linalg import (
    ablate_input,
    ablate_output,
    bake_direction,
    low_rank_scale,
    rank1,
    rank1_constrained,
    relative_norm,
)
from .schema import OPS, EditSpec

__all__ = ["EditResult", "execute_spec"]


@dataclass(frozen=True)
class EditResult:
    op: str
    module: str
    delta: np.ndarray
    relative_norm: float
    target_error: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op": self.op,
            "module": self.module,
            "delta_shape": list(self.delta.shape),
            "relative_norm": self.relative_norm,
            "target_error": self.target_error,
        }


def _need(name: str, value: Any) -> Any:
    if value is None:
        raise ValueError("op requires operand %r" % name)
    return value


def _target_error(W: np.ndarray, k: np.ndarray, v: np.ndarray) -> float:
    return float(np.linalg.norm(W @ np.asarray(k, dtype=np.float64) - v))


def execute_spec(
    spec: EditSpec,
    W: Any,
    *,
    k: Any = None,
    v: Any = None,
    direction: Any = None,
    a_ref: Any = None,
    alpha: Optional[float] = None,
    preserve: Any = None,
    donor_delta: Any = None,
    scale: float = 1.0,
) -> EditResult:
    """Run one validated :class:`EditSpec` against one weight matrix.

    Operand requirements by op:

    ``rank1`` / ``rank1_constrained``
        ``k`` and ``v``; the constrained op also accepts ``preserve``.
    ``bake_direction``
        ``a_ref``, ``direction`` and ``alpha``.
    ``ablate_output`` / ``ablate_input``
        ``direction``.
    ``low_rank_scale``
        ``donor_delta``; ``spec.rank`` and ``scale`` control the truncation.

    The ``solver`` field is a declaration, not a licence to train.  ``search``
    is refused; ``closed_form`` and ``donor`` are executable.
    """
    spec = spec.validate()
    if spec.op not in OPS:
        raise ValueError("unknown op %r" % spec.op)
    if spec.solver == "search":
        raise ValueError(
            "solver='search' is not executable; use closed_form or donor"
        )

    weight = np.asarray(W, dtype=np.float64)
    if weight.ndim != 2:
        raise ValueError("W must be 2-D (d_out, d_in)")

    target_error: Optional[float] = None
    if spec.op in ("rank1", "rank1_constrained"):
        key = np.asarray(_need("k", k), dtype=np.float64).reshape(-1)
        value = np.asarray(_need("v", v), dtype=np.float64).reshape(-1)
        if spec.op == "rank1":
            updated, delta = rank1(weight, key, value)
        else:
            updated, delta = rank1_constrained(
                weight,
                key,
                value,
                preserve=preserve,
                max_rel_norm=spec.constraints.max_rel_norm,
            )
        target_error = _target_error(updated, key, value)
    elif spec.op == "bake_direction":
        direction = np.asarray(_need("direction", direction), dtype=np.float64).reshape(-1)
        ref = np.asarray(_need("a_ref", a_ref), dtype=np.float64).reshape(-1)
        alpha = float(_need("alpha", alpha))
        updated, delta = bake_direction(weight, ref, direction, alpha)
    elif spec.op == "ablate_output":
        direction = np.asarray(_need("direction", direction), dtype=np.float64).reshape(-1)
        updated, delta = ablate_output(weight, direction)
    elif spec.op == "ablate_input":
        direction = np.asarray(_need("direction", direction), dtype=np.float64).reshape(-1)
        updated, delta = ablate_input(weight, direction)
    elif spec.op == "low_rank_scale":
        donor = np.asarray(_need("donor_delta", donor_delta), dtype=np.float64)
        if donor.shape != weight.shape:
            raise ValueError(
                "donor_delta shape %r does not match W %r"
                % (donor.shape, weight.shape)
            )
        delta = low_rank_scale(donor, spec.rank, scale=float(scale))
        updated = weight + delta
    else:  # pragma: no cover - kept for a future op added to OPS first
        raise ValueError("no executor for op %r" % spec.op)

    if delta.shape != weight.shape:
        raise ValueError("executor produced shape %r for W %r" % (delta.shape, weight.shape))
    if not np.isfinite(delta).all():
        raise ValueError("executor produced non-finite delta")
    rel = relative_norm(delta, weight)
    if rel > spec.constraints.max_rel_norm + 1e-12:
        raise ValueError(
            "executor produced relative norm %.6g above spec budget %.6g"
            % (rel, spec.constraints.max_rel_norm)
        )
    return EditResult(
        op=spec.op,
        module=spec.target.module,
        delta=delta,
        relative_norm=rel,
        target_error=target_error,
    )
