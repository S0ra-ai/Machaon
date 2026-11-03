"""Metrics that actually catch damage.

Accuracy hides small distribution shifts; KL does not.  Use KL for locality,
and always report quantiles rather than a mean.
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np


def kl_divergence(p, q, eps: float = 1e-12):
    """Per-sample KL(p || q). Accepts (n, vocab) or (vocab,)."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum(axis=-1, keepdims=True)
    q = q / q.sum(axis=-1, keepdims=True)
    return np.sum(p * (np.log(p) - np.log(q)), axis=-1)


def kl_summary(p_before, p_after) -> Dict[str, float]:
    kl = np.atleast_1d(kl_divergence(p_before, p_after))
    return {
        "kl_mean": float(kl.mean()),
        "kl_p50": float(np.quantile(kl, 0.50)),
        "kl_p95": float(np.quantile(kl, 0.95)),
        "kl_max": float(kl.max()),
    }


def perplexity_ratio(nll_before, nll_after) -> float:
    """exp(mean NLL_after - mean NLL_before); 1.0 means unchanged."""
    b = np.asarray(nll_before, dtype=np.float64)
    a = np.asarray(nll_after, dtype=np.float64)
    if b.shape != a.shape:
        raise ValueError("nll_before and nll_after must have the same shape")
    return float(np.exp(a.mean() - b.mean()))


def bootstrap_ci(values, stat: Callable = np.mean, n: int = 2000,
                 alpha: float = 0.05, seed: int = 0) -> Tuple[float, float]:
    """Percentile bootstrap interval. Report this, not a bare mean."""
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if v.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    rng = np.random.default_rng(seed)
    stats = np.empty(n, dtype=np.float64)
    for i in range(n):
        stats[i] = stat(rng.choice(v, size=v.size, replace=True))
    lo = float(np.quantile(stats, alpha / 2.0))
    hi = float(np.quantile(stats, 1.0 - alpha / 2.0))
    return lo, hi
