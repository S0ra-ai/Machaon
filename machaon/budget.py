"""Cumulative edit budget.

A single edit is cheap. Twenty of them are not: the damage compounds, and the
model degrades long before any individual edit looks unreasonable.

The budget tracks ``Σ‖ΔWᵢ‖ / ‖W‖`` per tensor and across the model. That sum is
an upper bound on ``‖ΣΔWᵢ‖ / ‖W‖`` by the triangle inequality, so it is
conservative: safe to enforce, never flattering.

Budget is charged at commit, not at apply, so a rolled-back transaction costs
nothing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np

__all__ = ["BudgetExceeded", "BudgetState", "EditBudget"]


class BudgetExceeded(RuntimeError):
    """Committing these deltas would exceed the declared budget."""


@dataclass(frozen=True)
class BudgetState:
    total: float
    limit: float | None
    remaining: float | None
    per_tensor: Dict[str, float] = field(default_factory=dict)
    tensor_limit: float | None = None

    @property
    def exhausted(self) -> bool:
        return self.limit is not None and self.total >= self.limit


class EditBudget:
    """A cap on cumulative relative change.

    ``max_total`` is the model-wide sum of per-tensor relative changes.
    ``max_per_tensor`` caps any single tensor.

    Empirical anchors from the benchmark results: eight rank-one edits at
    rel.Δ ≈ 0.013 each already cost 3–7% perplexity, so a total budget above
    ~0.15 is optimistic for a 124M–0.5B model.
    """

    def __init__(self, max_total: float | None = 0.10,
                 max_per_tensor: float | None = None) -> None:
        if max_total is not None and max_total <= 0:
            raise ValueError("max_total must be positive or None")
        if max_per_tensor is not None and max_per_tensor <= 0:
            raise ValueError("max_per_tensor must be positive or None")
        self.max_total = max_total
        self.max_per_tensor = max_per_tensor
        self._base: Dict[str, float] = {}
        self._spent: Dict[str, float] = {}

    # -- registration ---------------------------------------------------

    def register(self, name: str, weight) -> None:
        array = np.asarray(weight, dtype=np.float64)
        norm = float(np.linalg.norm(array))
        if norm == 0.0:
            raise ValueError("cannot budget a zero-norm tensor: %s" % name)
        self._base[name] = norm
        self._spent.setdefault(name, 0.0)

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(sorted(self._base))

    # -- accounting -----------------------------------------------------

    def relative(self, name: str, delta) -> float:
        if name not in self._base:
            raise KeyError("tensor %r is not registered with the budget" % name)
        return float(np.linalg.norm(np.asarray(delta, dtype=np.float64))) / self._base[name]

    def _prospective(self, deltas: Iterable[Tuple[str, object]]) -> Dict[str, float]:
        totals = dict(self._spent)
        for name, delta in deltas:
            totals[name] = totals.get(name, 0.0) + self.relative(name, delta)
        return totals

    def check(self, deltas: Iterable[Tuple[str, object]]) -> BudgetState:
        """Raise if committing ``deltas`` would exceed a limit. Changes nothing."""
        deltas = list(deltas)
        totals = self._prospective(deltas)
        for name, value in totals.items():
            if self.max_per_tensor is not None and value > self.max_per_tensor:
                raise BudgetExceeded(
                    "tensor %s would reach %.4f > per-tensor limit %.4f"
                    % (name, value, self.max_per_tensor)
                )
        total = sum(totals.values())
        if self.max_total is not None and total > self.max_total:
            raise BudgetExceeded(
                "cumulative relative change would reach %.4f > limit %.4f"
                % (total, self.max_total)
            )
        return self._state(total, totals)

    def add(self, deltas: Iterable[Tuple[str, object]]) -> BudgetState:
        """Check, then record. Raises before recording anything."""
        deltas = list(deltas)
        self.check(deltas)
        for name, delta in deltas:
            self._spent[name] = self._spent.get(name, 0.0) + self.relative(name, delta)
        return self.state()

    def state(self) -> BudgetState:
        return self._state(sum(self._spent.values()), self._spent)

    def _state(self, total: float, per_tensor: Mapping[str, float]) -> BudgetState:
        remaining = None if self.max_total is None else max(0.0, self.max_total - total)
        return BudgetState(
            total=total,
            limit=self.max_total,
            remaining=remaining,
            per_tensor=dict(sorted(per_tensor.items())),
            tensor_limit=self.max_per_tensor,
        )

    def __repr__(self) -> str:
        state = self.state()
        return "EditBudget(total=%.4f, limit=%s, tensors=%d)" % (
            state.total, state.limit, len(state.per_tensor))
