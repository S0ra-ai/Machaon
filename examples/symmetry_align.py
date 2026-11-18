"""Align permuted hidden units across the two matrices of an MLP layer.

    python examples/symmetry_align.py

No model download, no torch.  The example recovers a known hidden-unit
permutation and applies the same permutation to the outgoing matrix.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from machaon import (
    align_mlp_hidden,
    functional_alignment_check,
    merge_mlp_aligned,
)


def main() -> int:
    rng = np.random.default_rng(7)
    reference_in = rng.normal(size=(8, 6))
    reference_out = rng.normal(size=(4, 8))
    order = np.array([4, 0, 7, 2, 6, 1, 3, 5])
    candidate_in = reference_in[order]
    candidate_out = reference_out[:, order]

    aligned = align_mlp_hidden(
        reference_in, reference_out, candidate_in, candidate_out
    )
    print("permutation      ", list(aligned.units.permutation))
    print("mean similarity  ", "%.6f" % aligned.units.mean_similarity)
    print("min similarity   ", "%.6f" % aligned.units.min_similarity)
    print("incoming equal   ", bool(np.array_equal(aligned.incoming, reference_in)))
    print("outgoing equal   ", bool(np.array_equal(aligned.outgoing, reference_out)))
    merged = merge_mlp_aligned(
        reference_in, reference_out, candidate_in, candidate_out, alpha=0.5
    )
    print("merged mean sim  ", "%.6f" % merged.units.mean_similarity)
    probes = rng.normal(size=(16, reference_in.shape[1]))
    functional = functional_alignment_check(
        reference_in, reference_out, candidate_in, candidate_out, probes
    )
    print("function passed  ", functional["passed"])
    print("max output error ", "%.2e" % functional["max_abs_error"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
