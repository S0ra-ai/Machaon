"""Synthetic end-to-end demo: no model download, no GPU, no network.

It shows the only thing that matters: a plain rank-1 edit moves the protected
keys, a null-space-constrained edit does not, and the relative-norm budget is
the dial that trades target accuracy for safety.

The model here is a single linear layer, which is exactly what one MLP block
looks like from the edit's point of view.
"""
from __future__ import annotations

import sys
from typing import Dict, List

import numpy as np

from .linalg import rank1, rank1_constrained, relative_norm
from .verify import check_protected


def _quiet():
    """Apple Accelerate sets spurious FP flags on matmul; ignore on macOS only."""
    if sys.platform == "darwin":
        return np.errstate(all="ignore")
    return _NullContext()


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def run_demo(seed: int = 0, d_in: int = 64, d_out: int = 64,
             n_preserve: int = 32, n_probe: int = 256,
             max_rel_norm: float = 0.02) -> Dict[str, object]:
    def measure(label: str, W_new: np.ndarray, delta: np.ndarray) -> Dict[str, float]:
        target_err = float(np.linalg.norm(W_new @ k - v) / np.linalg.norm(v))
        preserve_drift = check_protected(W, delta, preserve)["max_drift"]
        probe_drift = np.linalg.norm((W_new - W) @ probes.T, axis=0) / np.maximum(
            np.linalg.norm(W @ probes.T, axis=0), 1e-12
        )
        return {
            "op": label,
            "rel_delta": relative_norm(delta, W),
            "target_err": target_err,
            "preserve_drift_max": preserve_drift,
            "probe_drift_p95": float(np.quantile(probe_drift, 0.95)),
        }

    with _quiet():
        rng = np.random.default_rng(seed)
        W = rng.normal(0.0, 0.5, size=(d_out, d_in))
        preserve = rng.normal(0.0, 1.0, size=(n_preserve, d_in))
        k = rng.normal(0.0, 1.0, size=(d_in,))
        v = W @ k + rng.normal(0.0, 1.5, size=(d_out,))
        probes = rng.normal(0.0, 1.0, size=(n_probe, d_in))

        W_naive, d_naive = rank1(W, k, v)
        W_safe, d_safe = rank1_constrained(W, k, v, preserve=preserve)
        W_capped, d_capped = rank1_constrained(
            W, k, v, preserve=preserve, max_rel_norm=max_rel_norm
        )

        rows: List[Dict[str, float]] = [
            measure("rank1 (naive)", W_naive, d_naive),
            measure("constrained (no cap)", W_safe, d_safe),
            measure("constrained (capped)", W_capped, d_capped),
        ]

    return {
        "config": {
            "d_in": d_in, "d_out": d_out,
            "n_preserve": n_preserve, "n_probe": n_probe,
            "max_rel_norm": max_rel_norm,
        },
        "rows": rows,
        "required_budget": rows[1]["rel_delta"],
    }


def main() -> None:
    report = run_demo()
    cfg = report["config"]
    print("Machaon demo — d_in=%d d_out=%d, %d protected keys, budget %.2f"
          % (cfg["d_in"], cfg["d_out"], cfg["n_preserve"], cfg["max_rel_norm"]))
    print("(a single linear layer, i.e. one MLP block seen from the edit's side)\n")

    header = "%-24s %11s %12s %16s %15s" % (
        "op", "rel.delta", "target err", "preserve drift", "probe drift p95")
    print(header)
    print("-" * len(header))
    for row in report["rows"]:
        print("%-24s %11.4f %12.2e %16.2e %15.4f" % (
            row["op"], row["rel_delta"], row["target_err"],
            row["preserve_drift_max"], row["probe_drift_p95"]))

    print("\nread it like this:")
    print("  target err     -> 0 means W'k == v, the edit landed")
    print("  preserve drift -> the naive edit moves protected keys; the")
    print("                    constrained edit leaves them at 1e-16")
    print("  rel.delta      -> how much of the layer you are spending")
    print()
    print("  this edit wants a budget of %.4f to land exactly."
          % report["required_budget"])
    print("  capped at %.2f, it spends %.2f and lands %.0f%% of the way."
          % (cfg["max_rel_norm"], cfg["max_rel_norm"],
             (1.0 - report["rows"][2]["target_err"]) * 100))
    print("  that trade -- exactness vs budget -- is the whole product.")


if __name__ == "__main__":
    main()
