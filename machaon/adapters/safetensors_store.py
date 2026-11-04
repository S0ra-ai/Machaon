"""Read and rewrite safetensors checkpoints without loading the whole model.

Supports a single ``.safetensors`` file, a directory holding one, and a
sharded checkpoint described by ``model.safetensors.index.json``.  Reads are
lazy: :meth:`Checkpoint.read` materialises one tensor at a time.

Writing is intentionally explicit.  :meth:`Checkpoint.write_with` produces a new
checkpoint and never mutates the source, because an edit you cannot undo is not
an edit.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

try:
    from safetensors import safe_open
    from safetensors.torch import save_file
except ImportError as exc:  # pragma: no cover - import-time dependency
    raise ImportError(
        "machaon.adapters needs safetensors and torch "
        "(pip install safetensors torch). The core maths needs neither."
    ) from exc


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: Tuple[int, ...]
    dtype: str


def _to_numpy(tensor):
    """NumPy has no bfloat16, so widen it exactly to float32 first."""
    import torch

    if tensor.dtype == torch.bfloat16:
        return tensor.float().numpy()
    return tensor.detach().cpu().numpy()


class Checkpoint:
    """A read-only view over a safetensors checkpoint."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self._weight_map: Optional[Dict[str, str]] = None
        self._single: Optional[Path] = None

        if self.path.is_dir():
            index = self.path / "model.safetensors.index.json"
            if index.exists():
                payload = json.loads(index.read_text())
                weight_map = payload.get("weight_map")
                if not isinstance(weight_map, dict) or not weight_map:
                    raise ValueError("index.json has no weight_map: %s" % index)
                self._weight_map = weight_map
            else:
                files = sorted(self.path.glob("*.safetensors"))
                if len(files) != 1:
                    raise ValueError(
                        "expected exactly one .safetensors file in %s, found %d"
                        % (self.path, len(files))
                    )
                self._single = files[0]
        elif self.path.suffix == ".safetensors":
            self._single = self.path
        else:
            raise ValueError("not a safetensors file or directory: %s" % self.path)

    # -- layout ---------------------------------------------------------

    @property
    def shards(self) -> List[Path]:
        if self._weight_map is not None:
            return sorted({self.path / name for name in self._weight_map.values()})
        assert self._single is not None
        return [self._single]

    def _shard_for(self, name: str) -> Path:
        if self._weight_map is not None:
            try:
                return self.path / self._weight_map[name]
            except KeyError:
                raise KeyError("unknown tensor: %s" % name) from None
        assert self._single is not None
        return self._single

    def names(self) -> List[str]:
        if self._weight_map is not None:
            return sorted(self._weight_map)
        assert self._single is not None
        with safe_open(str(self._single), framework="pt", device="cpu") as f:
            return sorted(f.keys())

    def _open(self, shard: Path):
        return safe_open(str(shard), framework="pt", device="cpu")

    # -- reads ----------------------------------------------------------

    def info(self, name: str) -> TensorInfo:
        with self._open(self._shard_for(name)) as f:
            slice_ = f.get_slice(name)
            return TensorInfo(name=name, shape=tuple(slice_.get_shape()),
                              dtype=str(slice_.get_dtype()))

    def read(self, name: str, dtype: Optional[Any] = np.float32) -> np.ndarray:
        """Materialise one tensor.  ``dtype=None`` keeps the stored dtype."""
        with self._open(self._shard_for(name)) as f:
            tensor = f.get_tensor(name)
        array = _to_numpy(tensor)
        return array if dtype is None else np.ascontiguousarray(array, dtype=dtype)

    def read_float64(self, name: str) -> np.ndarray:
        return self.read(name, dtype=np.float64)

    def summary(self) -> Dict[str, Any]:
        """Tensor count, parameter count and dtype mix, without loading data."""
        import math
        from collections import Counter

        dtypes: Counter = Counter()
        total = 0
        names = self.names()
        for name in names:
            info = self.info(name)
            count = int(math.prod(info.shape)) if info.shape else 1
            total += count
            dtypes[info.dtype] += count
        return {
            "tensors": len(names),
            "parameters": total,
            "dtypes": dict(dtypes),
            "shards": len(self.shards),
            "path": str(self.path),
        }

    def find(
        self,
        contains: Optional[str] = None,
        suffix: Optional[str] = None,
        layer: Optional[int] = None,
    ) -> List[str]:
        marker = None if layer is None else ".layers.%d." % int(layer)
        out = []
        for name in self.names():
            if contains and contains not in name:
                continue
            if suffix and not name.endswith(suffix):
                continue
            if marker and marker not in name:
                continue
            out.append(name)
        return out

    # -- writes ---------------------------------------------------------

    def write_tensor(self, name: str, values, out_dir: str | Path,
                     link_others: bool = True) -> Dict[str, Any]:
        """Rewrite only the shard that holds ``name``.

        Unchanged shards are hard-linked by default, so a 7B checkpoint costs
        one shard of memory and one shard of I/O instead of fifteen gigabytes
        of each. Falls back to a copy when linking is not possible.
        """
        import shutil

        import torch

        if name not in self.names():
            raise KeyError("unknown tensor: %s" % name)

        target = self._shard_for(name)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        if self._weight_map is not None:
            names = sorted(n for n, s in self._weight_map.items() if self.path / s == target)
        else:
            names = self.names()

        tensors: Dict[str, Any] = {}
        metadata = None
        with self._open(target) as f:
            metadata = f.metadata()
            for current in names:
                original = f.get_tensor(current)
                if current != name:
                    tensors[current] = original.contiguous()
                    continue
                array = np.ascontiguousarray(np.asarray(values), dtype=np.float32)
                expected = tuple(original.shape)
                if array.shape != expected:
                    raise ValueError(
                        "shape %r does not match %s %r" % (array.shape, name, expected))
                tensors[current] = torch.from_numpy(array).to(original.dtype).contiguous()

        destination = out / target.name
        save_file(tensors, str(destination), metadata=metadata)

        linked, copied = [], []
        for shard in self.shards:
            if shard == target:
                continue
            sibling = out / shard.name
            if sibling.exists():
                continue
            if link_others:
                try:
                    os.link(shard, sibling)
                    linked.append(sibling.name)
                    continue
                except OSError:
                    pass
            shutil.copy2(shard, sibling)
            copied.append(sibling.name)

        if self._weight_map is not None:
            (out / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": self._weight_map}, indent=2) + "\n")

        return {
            "tensor": name,
            "shard": target.name,
            "rewritten": destination.name,
            "linked": linked,
            "copied": copied,
        }

    def write_with(self, deltas: Mapping[str, np.ndarray], out_dir: str | Path) -> Dict[str, Any]:
        """Write a new checkpoint with ``deltas`` applied.  The source is untouched.

        Deltas are added in float64 and cast back to each tensor's original
        dtype.  Tensors without a delta are copied bit-for-bit.
        """
        import torch

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        known = set(self.names())
        unknown = [n for n in deltas if n not in known]
        if unknown:
            raise KeyError("deltas name unknown tensors: %s" % ", ".join(sorted(unknown)))

        report: Dict[str, Any] = {"tensors": {}, "files": []}
        for shard in self.shards:
            if self._weight_map is not None:
                names = sorted(n for n, s in self._weight_map.items() if self.path / s == shard)
                target = out / shard.name
            else:
                names = self.names()
                target = out / shard.name

            tensors: Dict[str, Any] = {}
            metadata = None
            with self._open(shard) as f:
                metadata = f.metadata()
                for name in names:
                    original = f.get_tensor(name)
                    if name not in deltas:
                        tensors[name] = original.contiguous()
                        continue
                    base = _to_numpy(original)
                    delta = np.asarray(deltas[name])
                    if delta.shape != base.shape:
                        raise ValueError(
                            "delta shape %r does not match %r for %s"
                            % (delta.shape, base.shape, name)
                        )
                    updated = base.astype(np.float64) + delta.astype(np.float64)
                    as_float32 = np.ascontiguousarray(updated, dtype=np.float32)
                    tensors[name] = torch.from_numpy(as_float32).to(original.dtype).contiguous()
                    denom = float(np.linalg.norm(base.astype(np.float64)))
                    report["tensors"][name] = {
                        "relative_change": float(
                            np.linalg.norm(delta.astype(np.float64)) / denom
                        ) if denom else None,
                        "changed_entries": int(np.count_nonzero(delta)),
                        "dtype": str(base.dtype),
                    }

            save_file(tensors, str(target), metadata=metadata)
            report["files"].append(str(target))

        if self._weight_map is not None:
            index = out / "model.safetensors.index.json"
            index.write_text(json.dumps({"weight_map": self._weight_map}, indent=2) + "\n")
            report["files"].append(str(index))

        return report
