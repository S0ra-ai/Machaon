"""Byte-compatible reader/writer for the OpenWSP edit format, version 1.

Specification: https://github.com/openwsp/openwsp/blob/main/docs/edit-format.md

An edit is a directory::

    edit/
    ├── manifest.json
    └── edit.npz        # t{i}_idx (int32), t{i}_val (uint16 bfloat16 bits)

The address is computed over *logical* arrays, never over the ZIP container, so
repacking preserves identity.  Every hash below is reproduced independently of
the reference implementation and checked against a golden fixture produced by
it (see ``tests/test_openwsp_interop.py``).

This module depends on NumPy only.  PyTorch tensors are accepted and converted
at the boundary.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np

FORMAT = "openwsp.edit"
FORMAT_VERSION = 1


class EditError(ValueError):
    """An edit is invalid, corrupted, or incompatible with its base."""


def _as_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):  # torch.Tensor, without importing torch
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def bf16_bits(values: Any) -> np.ndarray:
    """Round float32 to bfloat16 (nearest, ties to even) and return the bits.

    The reference implementation uses ``torch.Tensor.to(torch.bfloat16)``.  This
    is the same rounding expressed in integer arithmetic, so it needs no torch.
    """
    a = _as_array(values)
    if not np.issubdtype(a.dtype, np.floating):
        raise EditError("weights must be floating point")
    a = np.ascontiguousarray(a, dtype=np.float32)
    if not np.isfinite(a).all():
        raise EditError("weights must be finite")
    u = a.view(np.uint32).astype(np.uint64)
    u = u + 0x7FFF + ((u >> 16) & 1)
    bits = (u >> 16).astype("<u2")
    if not np.isfinite(bits_float32(bits)).all():
        raise EditError("weights overflow bfloat16")
    return bits


def bits_float32(bits: Any) -> np.ndarray:
    """Inverse of :func:`bf16_bits`: widen bfloat16 bits back to float32."""
    b = np.ascontiguousarray(bits, dtype="<u2")
    return (b.astype(np.uint32) << 16).view(np.float32)


def tensor_sha256(tensor: Any) -> str:
    """SHA-256 over contiguous little-endian bfloat16 bytes."""
    return hashlib.sha256(np.ascontiguousarray(bf16_bits(tensor), dtype="<u2").tobytes()).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()


def payload_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    """Digest of the logical payload, independent of ZIP timestamps."""
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        header = _canonical(
            {"name": name, "dtype": array.dtype.str, "shape": list(array.shape)}
        )
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(array.nbytes.to_bytes(8, "little"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def edit_sha256(manifest: Mapping[str, Any]) -> str:
    """Content address of the manifest, excluding the address field itself."""
    body = {k: v for k, v in manifest.items() if k != "edit_sha256"}
    return hashlib.sha256(_canonical(body)).hexdigest()


def export_edit(
    originals: Mapping[str, Any],
    edited: Mapping[str, Any],
    path: str | Path,
    *,
    base_model: str | None = None,
    revision: str | None = None,
) -> Dict[str, Any]:
    """Write a sparse v1 diff against the bf16-rounded base.

    Only tensors whose bf16 bits actually differ are recorded.  Entries that
    round back to the base value are absent from the diff by construction.
    """
    arrays: Dict[str, np.ndarray] = {}
    tensors = []

    if any(not isinstance(name, str) or not name for name in edited):
        raise EditError("tensor names must be nonempty strings")

    for name in sorted(edited):
        if name not in originals:
            raise EditError("unknown base tensor: %s" % name)
        base = bf16_bits(originals[name])
        changed = bf16_bits(edited[name])
        if base.shape != changed.shape:
            raise EditError("shape mismatch: %s" % name)
        indices = np.flatnonzero(base.reshape(-1) != changed.reshape(-1))
        if not len(indices):
            continue
        if base.size > np.iinfo(np.int32).max:
            raise EditError("tensor too large for int32 indices: %s" % name)

        i = len(tensors)
        arrays["t%d_idx" % i] = indices.astype("<i4")
        arrays["t%d_val" % i] = changed.reshape(-1)[indices].astype("<u2")

        b = bits_float32(base).astype(np.float64)
        e = bits_float32(changed).astype(np.float64)
        norm = float(np.linalg.norm(b))
        delta = float(np.linalg.norm(e - b))

        entry: Dict[str, Any] = {
            "name": name,
            "shape": list(base.shape),
            "dtype": "bfloat16",
            "form": "sparse",
            "changed_entries": int(len(indices)),
            "base_sha256": hashlib.sha256(np.ascontiguousarray(base).tobytes()).hexdigest(),
            "edited_sha256": hashlib.sha256(np.ascontiguousarray(changed).tobytes()).hexdigest(),
            "submitted_file": name + ".npy",
            "relative_change": (delta / norm) if norm else None,
        }
        if base.ndim == 2:
            rows, columns = np.unravel_index(indices, base.shape)
            entry["changed_rows"] = int(len(np.unique(rows)))
            entry["changed_columns"] = int(len(np.unique(columns)))
        tensors.append(entry)

    manifest: Dict[str, Any] = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "tensors": tensors,
        "changed_tensors": len(tensors),
        "changed_entries_total": sum(t["changed_entries"] for t in tensors),
        "payload_sha256": payload_sha256(arrays),
    }
    if base_model is not None:
        manifest["base_model"] = base_model
    if revision is not None:
        manifest["base_revision"] = revision
    manifest["edit_sha256"] = edit_sha256(manifest)

    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    # Publish the manifest last: a reader sees either the previous complete pair
    # or the new one, never a half-written edit.
    with tempfile.NamedTemporaryFile(dir=target, suffix=".npz", delete=False) as f:
        payload_tmp = Path(f.name)
        np.savez_compressed(f, **arrays)
    with tempfile.NamedTemporaryFile(mode="w", dir=target, suffix=".json", delete=False) as f:
        manifest_tmp = Path(f.name)
        json.dump(manifest, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
    try:
        os.replace(payload_tmp, target / "edit.npz")
        os.replace(manifest_tmp, target / "manifest.json")
    finally:
        payload_tmp.unlink(missing_ok=True)
        manifest_tmp.unlink(missing_ok=True)
    return manifest


def _checked_fixups(arrays: Mapping[str, np.ndarray], i: int, size: int):
    try:
        indices = arrays["t%d_idx" % i]
        values = arrays["t%d_val" % i]
    except KeyError:
        raise EditError("missing sparse payload for tensor %d" % i) from None
    if indices.ndim != 1 or indices.dtype.kind not in "iu":
        raise EditError("sparse indices must be a 1-D integer array")
    if values.ndim != 1 or values.dtype.kind != "u" or values.dtype.itemsize != 2:
        raise EditError("sparse values must be a 1-D uint16 array")
    if len(indices) != len(values):
        raise EditError("sparse indices and values differ in length")
    if len(indices) and (int(indices.min()) < 0 or int(indices.max()) >= size):
        raise EditError("sparse index outside its tensor")
    if len(np.unique(indices)) != len(indices):
        raise EditError("duplicate sparse indices")
    return indices.astype(np.int64, copy=False), values


def is_openwsp_edit(path: str | Path) -> bool:
    try:
        manifest = json.loads((Path(path) / "manifest.json").read_text())
    except (OSError, ValueError):
        return False
    return (
        isinstance(manifest, dict)
        and manifest.get("format") == FORMAT
        and manifest.get("format_version") == FORMAT_VERSION
    )


def read_edit(
    path: str | Path, originals: Mapping[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Validate every byte, then return ``(manifest, reconstructed float32)``.

    Nothing is mutated on failure.  Both content addresses and every per-tensor
    hash are checked, so a truncated or edited payload is refused rather than
    silently applied.
    """
    target = Path(path)
    try:
        manifest = json.loads((target / "manifest.json").read_text())
    except (OSError, ValueError) as exc:
        raise EditError("cannot read manifest: %s" % exc) from exc
    if not isinstance(manifest, dict):
        raise EditError("manifest must be an object")

    versioned = "format" in manifest or "format_version" in manifest
    if versioned and (
        manifest.get("format") != FORMAT
        or type(manifest.get("format_version")) is not int
        or manifest["format_version"] != FORMAT_VERSION
    ):
        raise EditError("unsupported edit format or version")
    if not isinstance(manifest.get("tensors"), list):
        raise EditError("manifest tensors must be a list")

    try:
        with np.load(target / "edit.npz", allow_pickle=False) as archive:
            if len(set(archive.files)) != len(archive.files):
                raise EditError("duplicate payload array names")
            arrays = {name: archive[name] for name in archive.files}
    except (OSError, ValueError) as exc:
        raise EditError("cannot read payload: %s" % exc) from exc

    if versioned:
        if manifest.get("payload_sha256") != payload_sha256(arrays):
            raise EditError("payload sha256 mismatch")
        if manifest.get("edit_sha256") != edit_sha256(manifest):
            raise EditError("edit address sha256 mismatch")

    rebuilt: Dict[str, np.ndarray] = {}
    for i, record in enumerate(manifest["tensors"]):
        if not isinstance(record, dict):
            raise EditError("tensor records must be objects")
        name = record.get("name")
        if not isinstance(name, str) or name not in originals:
            raise EditError("unknown tensor: %r" % (name,))
        if name in rebuilt:
            raise EditError("duplicate tensor: %s" % name)

        base = bf16_bits(originals[name])
        if record.get("dtype") != "bfloat16" or record.get("shape") != list(base.shape):
            raise EditError("shape or dtype mismatch: %s" % name)
        if record.get("base_sha256") != hashlib.sha256(
            np.ascontiguousarray(base).tobytes()
        ).hexdigest():
            raise EditError("base tensor sha256 mismatch: %s" % name)

        indices, values = _checked_fixups(arrays, i, base.size)
        form = record.get("form")
        if form == "lowrank":
            a = arrays.get("t%d_a" % i)
            b = arrays.get("t%d_b" % i)
            if (
                a is None
                or b is None
                or base.ndim != 2
                or a.ndim != 2
                or b.ndim != 2
                or a.shape[0] != base.shape[0]
                or b.shape[1] != base.shape[1]
                or a.shape[1] != b.shape[0]
                or a.dtype.kind != "f"
                or b.dtype.kind != "f"
                or not np.isfinite(a).all()
                or not np.isfinite(b).all()
            ):
                raise EditError("invalid low-rank factors: %s" % name)
            weights = bits_float32(base).astype(np.float64)
            for rank in range(a.shape[1]):
                weights = weights + np.outer(a[:, rank], b[rank])
            edited_bits = bf16_bits(weights.astype(np.float32)).reshape(-1)
        elif form == "sparse":
            edited_bits = base.copy().reshape(-1)
        else:
            raise EditError("unsupported edit form: %r" % (form,))

        edited_bits[indices] = values
        edited_bits = edited_bits.reshape(base.shape)
        if record.get("edited_sha256") != hashlib.sha256(
            np.ascontiguousarray(edited_bits).tobytes()
        ).hexdigest():
            raise EditError("edited tensor sha256 mismatch: %s" % name)

        rebuilt[name] = bits_float32(edited_bits)

    return manifest, rebuilt


def delta_from_edit(
    manifest: Mapping[str, Any],
    rebuilt: Mapping[str, Any],
    originals: Mapping[str, Any],
) -> Dict[str, np.ndarray]:
    """Per-tensor float32 delta, ready for :mod:`machaon.verify`."""
    out: Dict[str, np.ndarray] = {}
    for record in manifest["tensors"]:
        name = record["name"]
        base = bits_float32(bf16_bits(originals[name])).astype(np.float32)
        out[name] = np.asarray(rebuilt[name], dtype=np.float32) - base
    return out
