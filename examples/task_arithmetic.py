"""Extract and combine task vectors with a relative-norm budget.

    python examples/task_arithmetic.py

No model download. This is the arithmetic layer only; it does not claim that
the resulting direction improves a task.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from machaon import combine_task_vectors, task_vector


def main() -> int:
    rng = np.random.default_rng(11)
    base = rng.normal(size=(32, 24))
    tuned_a = base + 0.02 * rng.normal(size=base.shape)
    tuned_b = base + 0.03 * rng.normal(size=base.shape)
    a = task_vector(base, tuned_a)
    b = task_vector(base, tuned_b)

    print("task vector A rel.norm", "%.5f" % a.relative_norm)
    print("task vector B rel.norm", "%.5f" % b.relative_norm)
    for label, kwargs in (
        ("A + 0.5 B", {"coefficients": {"a": 1.0, "b": 0.5}}),
        ("clamped", {"coefficients": {"a": 1.0, "b": 0.5},
                     "max_rel_norm": 0.005, "clamp": True}),
    ):
        result = combine_task_vectors(
            {"a": a, "b": b}, base=base, **kwargs
        )
        print("%-14s rel.norm %.5f clamped=%s" % (
            label, result.relative_norm, result.clamped
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
