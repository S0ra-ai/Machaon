"""Run one constrained Python edit action in an isolated worker.

    python examples/action_edit.py

No model download, no torch.  The example shows a protected rank-one action,
its source hash and a rejected import.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from machaon import (
    ActionSpec,
    ActionValidationError,
    Target,
    execute_action,
    validate_action_source,
)


def main() -> int:
    rng = np.random.default_rng(7)
    W = rng.normal(0.0, 0.5, size=(32, 32))
    key = rng.normal(size=32)
    value = rng.normal(size=32)
    preserve = rng.normal(size=(4, 32))
    source = """
def edit(W):
    updated, delta = rank1_constrained(
        W, keys, values, preserve=preserve, max_rel_norm=max_rel_norm)
    return delta
"""
    spec = ActionSpec(
        source,
        Target("model.layers.0.mlp.down_proj.weight", 0),
        intent="demonstrate a protected action",
        max_rel_norm=0.2,
    ).validate()
    result = execute_action(
        spec, W, keys=key, values=value, preserve=preserve
    )
    drift = np.linalg.norm(result.delta @ preserve.T)
    print("source sha256     ", result.source_sha256)
    print("relative norm     ", "%.4f" % result.relative_norm)
    print("protected drift   ", "%.2e" % drift)
    print("protected verdict ", result.protected["passed"])
    print("worker elapsed ms ", "%.1f" % result.elapsed_ms)

    try:
        validate_action_source(
            "def edit(W):\n    import os\n    return W\n"
        )
    except ActionValidationError as exc:
        print("import rejected   ", str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
