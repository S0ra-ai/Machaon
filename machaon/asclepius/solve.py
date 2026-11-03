"""Value solvers. No gradients over model weights, ever."""
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np


def value_from_donor(donor_hidden) -> np.ndarray:
    """Take the residual state from a run that already produces the behaviour.

    One forward pass on a text that contains the fact, capture the hidden state
    at the matching position, flatten it.  This is the cheapest honest way to
    get ``v`` without optimising anything.
    """
    return np.asarray(donor_hidden, dtype=np.float64).reshape(-1).copy()


def refine_value(v0, score_fn: Callable[[np.ndarray], float], sigma: float = 0.05,
                 steps: int = 200, seed: int = 0, maximize: bool = True
                 ) -> Tuple[np.ndarray, float]:
    """(1+1) evolution strategy over ``v``.

    ``score_fn`` is any black-box function of the value vector (e.g. negative
    loss of the target token under a frozen model).  Memory cost is one vector;
    there is no backward pass and no optimiser state.
    """
    rng = np.random.default_rng(seed)
    best = np.asarray(v0, dtype=np.float64).copy()
    best_score = float(score_fn(best))
    for _ in range(int(steps)):
        candidate = best + rng.normal(0.0, sigma, size=best.shape)
        score = float(score_fn(candidate))
        better = score > best_score if maximize else score < best_score
        if better:
            best, best_score = candidate, score
    return best, best_score
