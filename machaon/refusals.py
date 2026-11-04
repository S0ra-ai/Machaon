"""Refusal rules for weight edits.

These are the checks a store must apply *before* accepting a tensor, not after.
They are adversarial by design: an edit that is merely well-formed can still be
a way of shrinking the model, or of corrupting a tensor that must not move.

The rules mirror the ones OpenWSP applies, so an edit accepted here is also
accepted there, and vice versa.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .interop.openwsp import bf16_bits


class EditRefused(ValueError):
    """An edit violated a rule that exists to protect the model."""


def _as_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def check_tensor(
    name: str,
    value: Any,
    original: Any,
    *,
    frozen: Iterable[str] = (),
    refuse_rescale: bool = True,
    rtol: float = 1e-5,
    atol: float = 1e-7,
) -> np.ndarray:
    """Validate one replacement tensor and return it as contiguous float32."""
    if not isinstance(name, str) or not name:
        raise EditRefused("tensor name must be a nonempty string")
    if name in set(frozen):
        raise EditRefused("frozen tensor: %s" % name)

    value = _as_array(value)
    if value.dtype.kind != "f":
        raise EditRefused("edit must be real floating point: %s" % name)

    base = np.ascontiguousarray(_as_array(original), dtype=np.float32)
    value = np.ascontiguousarray(value, dtype=np.float32)
    if value.shape != base.shape:
        raise EditRefused(
            "shape mismatch for %s: %r != %r" % (name, value.shape, base.shape)
        )
    if not np.isfinite(value).all():
        raise EditRefused("non-finite edit values: %s" % name)

    # Rounding to bfloat16 is what actually gets evaluated; a value that
    # overflows there would silently become inf.
    try:
        bf16_bits(value)
    except ValueError as exc:
        raise EditRefused("edit overflows bfloat16: %s (%s)" % (name, exc)) from exc

    if refuse_rescale and not np.array_equal(value, base):
        flat_base = base.reshape(-1).astype(np.float64)
        denom = float(flat_base @ flat_base)
        if denom > 0.0:
            scale = float(flat_base @ value.reshape(-1).astype(np.float64)) / denom
            if np.allclose(value, base * np.float32(scale), rtol=rtol, atol=atol):
                raise EditRefused(
                    "whole-tensor scalar rescaling is forbidden: %s (scale %.6g)"
                    % (name, scale)
                )
    return value
