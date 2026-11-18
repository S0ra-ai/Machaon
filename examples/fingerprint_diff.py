"""Fingerprint weights and report exactly where an edit changed them.

    python examples/fingerprint_diff.py

This is content detection, not cryptographic authentication.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from machaon import diff_tensors, model_fingerprint, tensor_fingerprint


def main() -> int:
    rng = np.random.default_rng(13)
    base = rng.normal(size=(16, 12))
    changed = base.copy()
    changed[2, 3] += 0.02
    changed[7, 1] -= 0.01

    before = tensor_fingerprint(base)
    after = tensor_fingerprint(changed)
    diff = diff_tensors(base, changed)
    model = model_fingerprint({"demo.weight": base})

    print("before sha256", before.sha256[:24])
    print("after sha256 ", after.sha256[:24])
    print("changed      ", diff["changed"])
    print("entries      ", diff["changed_entries"])
    print("support      ", "%.4f" % diff["support_fraction"])
    print("model sha256 ", model["sha256"][:24])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
