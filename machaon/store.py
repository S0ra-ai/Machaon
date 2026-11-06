"""Content-addressed delta store.

Only the delta is written, never the model.  Files are named by the SHA-256 of
their own bytes, so identical edits deduplicate and every read can be verified
against the address that names it.

Writes publish with ``os.replace`` and reads verify the digest by default.  A
store that silently returns bytes from a corrupted or swapped file would make
every downstream audit meaningless.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tempfile
from typing import Any, Dict, List

import numpy as np

__all__ = ["DeltaStore", "StoreCorrupted"]

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class StoreCorrupted(ValueError):
    """A stored delta does not match the address that names it."""


class DeltaStore:
    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)

    @staticmethod
    def _validate_sha(sha: str) -> str:
        if not isinstance(sha, str) or not _SHA256.match(sha):
            raise ValueError("not a SHA-256 address: %r" % (sha,))
        return sha

    def _path(self, sha: str) -> str:
        return os.path.join(self.root, sha[:2], sha + ".npy")

    def _mpath(self, sha: str) -> str:
        return os.path.join(self.root, sha[:2], sha + ".json")

    @staticmethod
    def _publish(path: str, raw: bytes) -> None:
        """Write atomically: a reader sees the old file or the complete new one."""
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def put(self, delta, manifest: Dict[str, Any]) -> str:
        buf = io.BytesIO()
        array = np.asarray(delta, dtype=np.float64)
        if not np.isfinite(array).all():
            raise ValueError("delta contains non-finite values")
        np.save(buf, array)
        raw = buf.getvalue()
        sha = hashlib.sha256(raw).hexdigest()
        path = self._path(sha)
        if not os.path.exists(path):
            self._publish(path, raw)
        record_path = self._mpath(sha)
        if not os.path.exists(record_path):
            record = dict(manifest)
            record["sha256"] = sha
            record["shape"] = list(array.shape)
            record["bytes"] = len(raw)
            payload = json.dumps(record, indent=2, sort_keys=True).encode()
            self._publish(record_path, payload)
        return sha

    def get(self, sha: str, verify: bool = True) -> np.ndarray:
        """Load a delta; by default prove its bytes still match the address."""
        sha = self._validate_sha(sha)
        path = self._path(sha)
        if not os.path.exists(path):
            raise KeyError("unknown delta: %s" % sha)
        with open(path, "rb") as handle:
            raw = handle.read()
        if verify:
            actual = hashlib.sha256(raw).hexdigest()
            if actual != sha:
                raise StoreCorrupted(
                    "delta %s is corrupted: file digest is %s" % (sha, actual)
                )
        try:
            with io.BytesIO(raw) as buf:
                return np.load(buf, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise StoreCorrupted("delta %s cannot be decoded: %s" % (sha, exc)) from exc

    def manifest(self, sha: str) -> Dict[str, Any]:
        sha = self._validate_sha(sha)
        with open(self._mpath(sha)) as f:
            return json.load(f)

    def list(self) -> List[str]:
        out = []
        for _dirpath, _, files in os.walk(self.root):
            for name in files:
                if name.endswith(".npy"):
                    out.append(name[:-4])
        return sorted(out)

    def __contains__(self, sha: str) -> bool:
        try:
            return os.path.exists(self._path(self._validate_sha(sha)))
        except ValueError:
            return False
