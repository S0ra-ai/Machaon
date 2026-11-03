"""Core linear-algebra primitives for weight surgery.

Every function is pure: arrays in, ``(new_weight, delta)`` out.  No gradients,
no training loops, no framework types.  Internally everything is float64;
callers cast back to their own dtype after the edit.

Conventions
-----------
W      : (d_out, d_in) weight matrix
k      : (d_in,)       key   -- the input the edit must memorise
v      : (d_out,)      value -- the output the edit must produce
d      : (d_out,) or (d_in,) unit direction
delta  : (d_out, d_in) additive update, ``W_new = W + delta``
"""
from __future__ import annotations

import functools
import sys

import numpy as np

__all__ = [
    "relative_norm",
    "clamp_rel_norm",
    "nullspace_basis",
    "nullspace_projector",
    "project_complement",
    "rank1",
    "rank1_constrained",
    "bake_direction",
    "ablate_output",
    "ablate_input",
    "low_rank_scale",
]


def _quiet_blas(fn):
    """Silence Apple Accelerate's spurious matmul FP flags on macOS.

    Accelerate leaves divide-by-zero / overflow / invalid flags set after a
    matmul, which numpy reports as RuntimeWarning on every later call even
    though the values are exact (verified in tests).  We suppress only that
    platform quirk, only inside our own kernels; on other platforms numpy's
    error reporting is untouched.
    """
    if sys.platform != "darwin":
        return fn

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with np.errstate(all="ignore"):
            return fn(*args, **kwargs)

    return wrapper


def _as_weight(W):
    W = np.asarray(W, dtype=np.float64)
    if W.ndim != 2:
        raise ValueError("W must be 2-D (d_out, d_in), got shape %r" % (W.shape,))
    return W


def _as_vec(x, name, size=None):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if size is not None and x.size != size:
        raise ValueError("%s has size %d, expected %d" % (name, x.size, size))
    return x


@_quiet_blas
def relative_norm(delta, W):
    """||delta||_F / ||W||_F -- the only budget that matters for safety."""
    W = _as_weight(W)
    denom = float(np.linalg.norm(W))
    if denom == 0.0:
        raise ValueError("reference weight has zero Frobenius norm")
    return float(np.linalg.norm(np.asarray(delta, dtype=np.float64)) / denom)


def clamp_rel_norm(delta, W, max_rel_norm):
    """Scale ``delta`` down until it fits the relative-norm budget.

    Note: clamping breaks the exact ``W' k == v`` guarantee.  The returned
    ``rel`` is the *post-clamp* relative norm.
    """
    delta = np.asarray(delta, dtype=np.float64)
    if max_rel_norm <= 0:
        raise ValueError("max_rel_norm must be positive")
    rel = relative_norm(delta, W)
    if rel <= max_rel_norm:
        return delta, rel
    return delta * (max_rel_norm / rel), max_rel_norm


@_quiet_blas
def nullspace_basis(preserve, rtol=1e-10):
    """Orthonormal basis ``Q`` (d_in, r) of ``span(preserve)``.

    Computed through SVD rather than a normal-equation inverse, so it stays
    well conditioned when keys are nearly collinear.  Working with the basis
    instead of a d_in x d_in projector keeps memory at O(d_in * r): for a
    4096-wide layer that is 0.5 MB instead of 128 MB.
    """
    K = np.atleast_2d(np.asarray(preserve, dtype=np.float64))
    if K.ndim != 2:
        raise ValueError("preserve must be (n_keys, d_in)")
    d = K.shape[1]
    if K.shape[0] == 0:
        return np.zeros((d, 0))
    _, S, Vt = np.linalg.svd(K, full_matrices=False)
    if S.size == 0 or S[0] == 0.0:
        return np.zeros((d, 0))
    rank = int((S > S[0] * rtol).sum())
    Q = Vt[:rank].T
    return np.ascontiguousarray(Q)


@_quiet_blas
def project_complement(k, Q):
    """Remove from ``k`` every component lying in ``span(Q)``.

    ``out = k - Q Q^T k``, evaluated column by column so the cost is O(d_in * r)
    and no large intermediate matrix is allocated.
    """
    k = _as_vec(k, "k")
    Q = np.asarray(Q, dtype=np.float64)
    if Q.ndim != 2:
        raise ValueError("Q must be (d_in, r)")
    if Q.shape[0] != k.size:
        raise ValueError(
            "basis dimension %d does not match key size %d" % (Q.shape[0], k.size)
        )
    out = k.copy()
    for j in range(Q.shape[1]):
        q = Q[:, j]
        out -= q * float((q * k).sum())
    return out


@_quiet_blas
def nullspace_projector(preserve, rtol=1e-10):
    """Dense projector ``I - Q Q^T`` (d_in x d_in).

    Convenience for inspection and tests; prefer :func:`project_complement`
    when you only need to project a few vectors.
    """
    K = np.atleast_2d(np.asarray(preserve, dtype=np.float64))
    if K.ndim != 2:
        raise ValueError("preserve must be (n_keys, d_in)")
    Q = nullspace_basis(K, rtol)
    P = np.eye(K.shape[1])
    for j in range(Q.shape[1]):
        q = Q[:, j]
        P -= np.outer(q, q)
    return P


@_quiet_blas
def rank1(W, k, v):
    """Classic rank-one edit (ROME).

        delta = (v - W k) k^T / (k^T k)

    Makes ``W' k == v`` exactly; says nothing about any other key.
    """
    W = _as_weight(W)
    k = _as_vec(k, "k", W.shape[1])
    v = _as_vec(v, "v", W.shape[0])
    denom = float(k @ k)
    if denom < 1e-12:
        raise ValueError("key has negligible norm")
    delta = np.outer(v - W @ k, k) / denom
    return W + delta, delta


@_quiet_blas
def rank1_constrained(W, k, v, preserve=None, max_rel_norm=None, eps=1e-12):
    """Rank-one edit that leaves a protected key set untouched.

        k_perp = P k,  P = I - Q Q^T
        delta  = (v - W k) k_perp^T / (k_perp^T k)

    Because ``k_perp`` is orthogonal to every protected key, ``delta @ k_i == 0``
    to machine precision: the protected behaviour is unchanged to first order.
    This is the whole trick, and the reason edits do not destroy the model.

    Raises if the target key lives inside the protected subspace -- such an edit
    is not expressible without violating the constraint, and silently
    proceeding would be a lie.
    """
    W = _as_weight(W)
    k = _as_vec(k, "k", W.shape[1])
    v = _as_vec(v, "v", W.shape[0])

    k_perp = k
    if preserve is not None:
        Q = nullspace_basis(preserve)
        k_perp = project_complement(k, Q)
        kn = float(np.linalg.norm(k))
        if np.linalg.norm(k_perp) <= eps * max(kn, 1.0):
            raise ValueError(
                "target key lies in the preserved subspace: the edit cannot be "
                "made without changing protected behaviour"
            )

    denom = float(k_perp @ k)
    if abs(denom) < eps:
        raise ValueError("degenerate key geometry (k_perp^T k ~ 0)")

    delta = np.outer(v - W @ k, k_perp) / denom
    if max_rel_norm is not None:
        delta, _ = clamp_rel_norm(delta, W, max_rel_norm)
    return W + delta, delta


@_quiet_blas
def bake_direction(W, a_ref, d, alpha):
    """Bake an inference-time steering vector into the weights.

        delta = alpha * d a_ref^T / (a_ref^T a_ref)

    After the edit, ``W' a_ref == W a_ref + alpha * d`` -- exactly the effect of
    adding ``alpha * d`` to that module's output, but permanently and with no
    runtime hook.
    """
    W = _as_weight(W)
    a = _as_vec(a_ref, "a_ref", W.shape[1])
    d = _as_vec(d, "d", W.shape[0])
    n = float(np.linalg.norm(d))
    if n == 0.0:
        raise ValueError("direction d has zero norm")
    d = d / n
    denom = float(a @ a)
    if denom < 1e-12:
        raise ValueError("reference activation has negligible norm")
    delta = alpha * np.outer(d, a) / denom
    return W + delta, delta


@_quiet_blas
def ablate_output(W, d):
    """Remove the component of the *output* along ``d``.

        W' = (I - d d^T) W
    """
    W = _as_weight(W)
    d = _as_vec(d, "d", W.shape[0])
    n = float(np.linalg.norm(d))
    if n == 0.0:
        raise ValueError("direction d has zero norm")
    d = d / n
    delta = -np.outer(d, d @ W)
    return W + delta, delta


@_quiet_blas
def ablate_input(W, d):
    """Remove the component of the *input* along ``d``.

        W' = W (I - d d^T)
    """
    W = _as_weight(W)
    d = _as_vec(d, "d", W.shape[1])
    n = float(np.linalg.norm(d))
    if n == 0.0:
        raise ValueError("direction d has zero norm")
    d = d / n
    delta = -np.outer(W @ d, d)
    return W + delta, delta


@_quiet_blas
def low_rank_scale(delta, rank, scale=1.0):
    """Truncate ``delta`` to rank ``rank`` and rescale it.

    Useful for compressing an edit before storing it, or for dialling a
    capability down rather than off.
    """
    delta = np.asarray(delta, dtype=np.float64)
    if delta.ndim != 2:
        raise ValueError("delta must be 2-D")
    if rank < 1:
        raise ValueError("rank must be >= 1")
    U, S, Vt = np.linalg.svd(delta, full_matrices=False)
    r = min(int(rank), S.size)
    return (U[:, :r] * (S[:r] * float(scale))) @ Vt[:r, :]
