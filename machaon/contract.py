"""Budget-aware synthesis of a protected-key contract.

The projection in :mod:`machaon.linalg` is exact for a fixed protected set.
In practice the protected set is often larger than a strict relative-norm
budget can certify. This module makes that trade-off explicit: it admits
candidate keys while the exact target-preserving update remains inside the
budget, and reports the keys that could not be admitted.

The selection rule is greedy. For equal weights it chooses the candidate that
adds the least relative norm at each step. For weighted keys it maximises
weight per unit of added norm. Every returned update is exact for the target
and every selected key; rejected keys are never silently described as
protected.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .budget import BudgetExceeded
from .linalg import rank1, rank1_constrained, relative_norm

__all__ = ["ContractSolution", "synthesize_contract"]


@dataclass(frozen=True)
class ContractSolution:
    """A certified update plus the protected set it actually covers."""

    delta: np.ndarray
    selected: Tuple[int, ...]
    rejected: Tuple[int, ...]
    relative_norm: float
    target_error: float
    exact: bool = True


def _residual(W: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    return v - W @ k


def _solve(W, k, v, keys):
    if keys is None or len(keys) == 0:
        return rank1(W, k, v)
    return rank1_constrained(W, k, v, preserve=keys)


def synthesize_contract(
    W,
    k,
    v,
    candidates,
    max_rel_norm: float,
    weights: Optional[Sequence[float]] = None,
) -> ContractSolution:
    """Select the largest budget-feasible certified protected set.

    Parameters
    ----------
    W, k, v:
        The weight matrix, target key and desired value.
    candidates:
        ``(n, d_in)`` array of keys that may be protected.
    max_rel_norm:
        Relative Frobenius budget for the exact update.
    weights:
        Optional nonnegative priority for each candidate. With equal weights
        the greedy rule minimises the norm added at each admission.

    Raises
    ------
    BudgetExceeded
        If even the unconstrained rank-one update exceeds the budget.
    """
    W = np.asarray(W, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64).reshape(-1)
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    candidates = np.asarray(candidates, dtype=np.float64)
    if candidates.ndim != 2 or candidates.shape[1] != W.shape[1]:
        raise ValueError("candidates must be (n_keys, d_in)")
    if not np.isfinite(max_rel_norm) or max_rel_norm <= 0:
        raise ValueError("max_rel_norm must be positive")

    n = candidates.shape[0]
    if weights is None:
        weights = np.ones(n, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if weights.size != n:
            raise ValueError("weights must match the number of candidates")
        if not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("weights must be finite and nonnegative")

    _, base_delta = _solve(W, k, v, None)
    base_rel = relative_norm(base_delta, W)
    if base_rel > max_rel_norm:
        raise BudgetExceeded(
            "unprotected edit needs %.6f > budget %.6f; increase the budget "
            "or change the target" % (base_rel, max_rel_norm)
        )

    selected = []
    available = list(range(n))
    delta = base_delta
    current_rel = base_rel

    while available:
        best = None
        best_score = None
        for index in available:
            trial = selected + [index]
            try:
                _, trial_delta = _solve(W, k, v, candidates[trial])
            except ValueError:
                continue
            trial_rel = relative_norm(trial_delta, W)
            if trial_rel > max_rel_norm:
                continue
            left = weights[index]
            added = max(trial_rel - current_rel, 0.0)
            score = float(left) / max(added, 1e-15)
            key = (score, -trial_rel, -index)
            if best_score is None or key > best_score:
                best_score = key
                best = (index, trial_delta, trial_rel)
        if best is None:
            break
        index, delta, current_rel = best
        selected.append(index)
        available.remove(index)

    selected_set = set(selected)
    rejected = tuple(i for i in range(n) if i not in selected_set)
    W_new = W + delta
    target_error = float(np.linalg.norm(W_new @ k - v) / max(np.linalg.norm(v), 1e-12))
    return ContractSolution(
        delta=np.asarray(delta, dtype=np.float64),
        selected=tuple(selected),
        rejected=rejected,
        relative_norm=float(current_rel),
        target_error=target_error,
        exact=True,
    )
