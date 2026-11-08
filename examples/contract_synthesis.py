"""Show what fits when the full protected set exceeds a budget.

    python examples/contract_synthesis.py

No model download, no torch. The example prints the admitted subset, the
rejected candidates, and a post-hoc protected-key audit.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from machaon import synthesize_contract
from machaon.verify import check_protected


def main() -> int:
    rng = np.random.default_rng(7)
    W = rng.normal(0.0, 0.5, size=(64, 64))
    k = rng.normal(0.0, 1.0, size=64)
    with np.errstate(all="ignore"):
        v = W @ k + rng.normal(0.0, 1.5, size=64)
    candidates = rng.normal(0.0, 1.0, size=(32, 64))

    print("budget  admitted  rejected  rel.delta  target err  protected drift")
    print("-" * 76)
    for budget in (0.05, 0.06, 0.07):
        solution = synthesize_contract(
            W, k, v, candidates, max_rel_norm=budget)
        selected = candidates[list(solution.selected)]
        check = (
            check_protected(W, solution.delta, selected)
            if len(selected) else {"max_drift": 0.0, "passed": True}
        )
        print(
            "%-6.2f %8d %9d %10.4f %11.2e %16.2e" % (
                budget,
                len(solution.selected),
                len(solution.rejected),
                solution.relative_norm,
                solution.target_error,
                check["max_drift"],
            )
        )
        print("  selected:", list(solution.selected))
        print("  rejected:", list(solution.rejected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
