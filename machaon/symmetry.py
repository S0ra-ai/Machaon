"""Weight-space symmetry helpers.

Independent solutions often describe the same function in a different unit
order.  This module provides a deliberately small, NumPy-only alignment layer:
cosine matching for hidden units, explicit row/column permutations, and a
matched pair operation for the two matrices around an MLP hidden layer.

Only permutation and positive scaling are handled.  Rotations, attention-head
permutations, cross-architecture alignment and full linear mode connectivity
are outside this module.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from ._fpe import quiet_blas

__all__ = [
    "UnitAlignment",
    "MLPAlignment",
    "permute_rows",
    "permute_columns",
    "align_rows",
    "align_mlp_hidden",
    "merge_mlp_aligned",
    "mlp_outputs",
    "functional_alignment_check",
]


@dataclass(frozen=True)
class UnitAlignment:
    permutation: Tuple[int, ...]
    aligned: np.ndarray
    similarities: Tuple[float, ...]
    mean_similarity: float
    min_similarity: float
    passed: bool
    metadata: dict

    def to_dict(self) -> dict:
        return {
            "permutation": list(self.permutation),
            "aligned_shape": list(self.aligned.shape),
            "similarities": list(self.similarities),
            "mean_similarity": self.mean_similarity,
            "min_similarity": self.min_similarity,
            "passed": self.passed,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class MLPAlignment:
    incoming: np.ndarray
    outgoing: np.ndarray
    units: UnitAlignment

    def to_dict(self) -> dict:
        return {
            "incoming_shape": list(self.incoming.shape),
            "outgoing_shape": list(self.outgoing.shape),
            "units": self.units.to_dict(),
        }


@dataclass(frozen=True)
class MergedMLP:
    incoming: np.ndarray
    outgoing: np.ndarray
    units: UnitAlignment
    alpha: float

    def to_dict(self) -> dict:
        return {
            "incoming_shape": list(self.incoming.shape),
            "outgoing_shape": list(self.outgoing.shape),
            "alpha": self.alpha,
            "units": self.units.to_dict(),
        }


def _matrix(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("%s must be 2-D" % name)
    if not np.isfinite(array).all():
        raise ValueError("%s contains non-finite values" % name)
    return array


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1)
    safe = np.where(norms > 0.0, norms, 1.0)
    return matrix / safe[:, None], norms


def permute_rows(matrix, permutation: Sequence[int]) -> np.ndarray:
    """Return rows reordered by a one-to-one permutation."""
    array = _matrix(matrix, "matrix")
    order = np.asarray(permutation, dtype=np.int64)
    if order.ndim != 1 or order.size != array.shape[0]:
        raise ValueError("permutation must have one index per row")
    if sorted(int(i) for i in order) != list(range(array.shape[0])):
        raise ValueError("permutation must be one-to-one and cover every row")
    return np.ascontiguousarray(array[order])


def permute_columns(matrix, permutation: Sequence[int]) -> np.ndarray:
    """Return columns reordered by a one-to-one permutation."""
    array = _matrix(matrix, "matrix")
    order = np.asarray(permutation, dtype=np.int64)
    if order.ndim != 1 or order.size != array.shape[1]:
        raise ValueError("permutation must have one index per column")
    if sorted(int(i) for i in order) != list(range(array.shape[1])):
        raise ValueError("permutation must be one-to-one and cover every column")
    return np.ascontiguousarray(array[:, order])


def _greedy_match(scores: np.ndarray) -> np.ndarray:
    """Confidence-first one-to-one matching; ties resolve by candidate index."""
    n = scores.shape[0]
    remaining = set(range(n))
    order = sorted(range(n), key=lambda i: (-float(np.max(scores[i])), i))
    permutation = [-1] * n
    for ref_index in order:
        available = list(remaining)
        if not available:
            raise ValueError("no candidate units remain")
        best = min(
            available,
            key=lambda j: (-float(scores[ref_index, j]), j),
        )
        permutation[ref_index] = best
        remaining.remove(best)
    return np.asarray(permutation, dtype=np.int64)


@quiet_blas
def align_rows(
    reference,
    candidate,
    *,
    axis: int = 0,
    min_similarity: float = 0.0,
    max_units: int = 4096,
) -> UnitAlignment:
    """Align candidate rows/columns to reference units by cosine similarity.

    ``permutation[i]`` is the candidate unit aligned to reference unit ``i``.
    Matching is greedy but confidence-first, not a full Hungarian solve; the
    result records that caveat and all similarities.
    """
    ref = _matrix(reference, "reference")
    cand = _matrix(candidate, "candidate")
    if ref.shape != cand.shape:
        raise ValueError("reference and candidate must have the same shape")
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1")
    if not math.isfinite(float(min_similarity)):
        raise ValueError("min_similarity must be finite")
    units = ref.shape[0] if axis == 0 else ref.shape[1]
    if units > int(max_units):
        raise ValueError(
            "alignment has %d units, above max_units=%d" % (units, max_units)
        )

    original_cand = cand
    if axis == 1:
        ref = ref.T
        cand = cand.T
    ref_rows, ref_norms = _unit_rows(ref)
    cand_rows, cand_norms = _unit_rows(cand)
    scores = ref_rows @ cand_rows.T
    scores[ref_norms == 0.0, :] = 0.0
    scores[:, cand_norms == 0.0] = 0.0
    permutation = _greedy_match(scores)
    similarities = scores[np.arange(units), permutation]
    aligned = (
        np.ascontiguousarray(original_cand[permutation])
        if axis == 0
        else np.ascontiguousarray(original_cand[:, permutation])
    )
    zero_units = int((ref_norms == 0.0).sum() + (cand_norms == 0.0).sum())
    passed = bool(np.all(similarities >= float(min_similarity)))
    return UnitAlignment(
        permutation=tuple(int(i) for i in permutation),
        aligned=aligned,
        similarities=tuple(float(value) for value in similarities),
        mean_similarity=float(np.mean(similarities)) if units else 0.0,
        min_similarity=float(np.min(similarities)) if units else 0.0,
        passed=passed,
        metadata={
            "axis": int(axis),
            "zero_units": zero_units,
            "matching": "greedy_confidence_first",
            "optimal": False,
        },
    )


def align_mlp_hidden(
    reference_in,
    reference_out,
    candidate_in,
    candidate_out,
    *,
    min_similarity: float = 0.0,
    max_units: int = 4096,
) -> MLPAlignment:
    """Align both matrices around one hidden layer consistently.

    ``reference_in`` / ``candidate_in`` are ``(hidden, d_in)`` and
    ``reference_out`` / ``candidate_out`` are ``(d_out, hidden)``.  The same
    hidden-unit permutation is applied to candidate rows and outgoing columns.
    """
    ref_in = _matrix(reference_in, "reference_in")
    ref_out = _matrix(reference_out, "reference_out")
    cand_in = _matrix(candidate_in, "candidate_in")
    cand_out = _matrix(candidate_out, "candidate_out")
    if ref_in.shape != cand_in.shape:
        raise ValueError("incoming matrices must have the same shape")
    if ref_out.shape != cand_out.shape:
        raise ValueError("outgoing matrices must have the same shape")
    if ref_in.shape[0] != ref_out.shape[1]:
        raise ValueError("reference hidden dimensions do not agree")
    if cand_in.shape[0] != cand_out.shape[1]:
        raise ValueError("candidate hidden dimensions do not agree")
    aligned_in = align_rows(
        ref_in, cand_in, min_similarity=min_similarity, max_units=max_units
    )
    incoming = aligned_in.aligned
    outgoing = permute_columns(cand_out, aligned_in.permutation)
    return MLPAlignment(incoming=incoming, outgoing=outgoing, units=aligned_in)


def merge_mlp_aligned(
    reference_in,
    reference_out,
    candidate_in,
    candidate_out,
    *,
    alpha: float = 0.5,
    min_similarity: float = 0.0,
    max_units: int = 4096,
) -> MergedMLP:
    """Permutation-align a candidate MLP, then linearly interpolate it.

    ``alpha=0`` returns the reference matrices; ``alpha=1`` returns the
    candidate after alignment.  The same hidden-unit permutation is applied to
    both matrices before averaging.
    """
    alpha = float(alpha)
    if not math.isfinite(alpha) or not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1]")
    ref_in = _matrix(reference_in, "reference_in")
    ref_out = _matrix(reference_out, "reference_out")
    alignment = align_mlp_hidden(
        ref_in,
        ref_out,
        candidate_in,
        candidate_out,
        min_similarity=min_similarity,
        max_units=max_units,
    )
    incoming = (1.0 - alpha) * ref_in + alpha * alignment.incoming
    outgoing = (1.0 - alpha) * ref_out + alpha * alignment.outgoing
    return MergedMLP(
        incoming=incoming,
        outgoing=outgoing,
        units=alignment.units,
        alpha=alpha,
    )


@quiet_blas
def mlp_outputs(
    incoming,
    outgoing,
    inputs,
    *,
    activation: str = "relu",
) -> np.ndarray:
    """Evaluate a one-hidden-layer MLP on probe inputs."""
    incoming = _matrix(incoming, "incoming")
    outgoing = _matrix(outgoing, "outgoing")
    probes = np.asarray(inputs, dtype=np.float64)
    if probes.ndim != 2 or probes.shape[1] != incoming.shape[1]:
        raise ValueError("inputs must be (n, d_in)")
    if incoming.shape[0] != outgoing.shape[1]:
        raise ValueError("incoming hidden width does not match outgoing")
    if not np.isfinite(probes).all():
        raise ValueError("inputs contain non-finite values")
    hidden = probes @ incoming.T
    if activation == "relu":
        hidden = np.maximum(hidden, 0.0)
    elif activation != "linear":
        raise ValueError("activation must be 'relu' or 'linear'")
    return hidden @ outgoing.T


def functional_alignment_check(
    reference_in,
    reference_out,
    candidate_in,
    candidate_out,
    probes,
    *,
    activation: str = "relu",
    min_similarity: float = 0.0,
    atol: float = 1e-9,
    max_units: int = 4096,
) -> dict:
    """Compare aligned MLP outputs on probes, not just unit coordinates."""
    if not math.isfinite(float(atol)) or atol < 0.0:
        raise ValueError("atol must be finite and non-negative")
    alignment = align_mlp_hidden(
        reference_in,
        reference_out,
        candidate_in,
        candidate_out,
        min_similarity=min_similarity,
        max_units=max_units,
    )
    reference_outputs = mlp_outputs(
        reference_in, reference_out, probes, activation=activation
    )
    aligned_outputs = mlp_outputs(
        alignment.incoming,
        alignment.outgoing,
        probes,
        activation=activation,
    )
    max_abs = float(np.max(np.abs(reference_outputs - aligned_outputs)))
    scale = float(np.linalg.norm(reference_outputs))
    relative = max_abs / scale if scale > 0.0 else None
    return {
        "passed": bool(alignment.units.passed and max_abs <= float(atol)),
        "max_abs_error": max_abs,
        "relative_error": relative,
        "atol": float(atol),
        "activation": activation,
        "n_probes": int(reference_outputs.shape[0]),
        "shape": list(reference_outputs.shape),
        "units": alignment.units.to_dict(),
    }
