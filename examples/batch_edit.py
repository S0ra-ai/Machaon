"""Batch many key/value pairs into one delta.

    python examples/batch_edit.py

No model download, no torch.  The example shows the exact batch, the explicit
clamp, and the protected-key projection.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from machaon import solve_batch


def main() -> int:
    rng = np.random.default_rng(7)
    W = rng.normal(0.0, 0.5, size=(64, 64))
    keys = rng.normal(0.0, 1.0, size=(6, 64))
    values = rng.normal(0.0, 1.0, size=(6, 64))
    preserve = rng.normal(0.0, 1.0, size=(4, 64))

    print("case                 rel.delta   target err  exact  clamped  rank")
    print("-" * 72)
    for label, kwargs in (
        ("exact", {}),
        ("protected", {"preserve": preserve}),
        ("clamped 0.05", {"max_rel_norm": 0.05}),
    ):
        result = solve_batch(W, keys, values, **kwargs)
        print("%-18s %10.4f %11.2e %6s %8s %5d" % (
            label,
            result.relative_norm,
            result.target_error,
            result.exact,
            result.clamped,
            result.rank,
        ))
        if "preserve" in kwargs:
            with np.errstate(all="ignore"):
                drift = np.linalg.norm(result.delta @ preserve.T)
            print("  protected drift: %.2e" % drift)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
