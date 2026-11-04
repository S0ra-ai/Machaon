"""Content-addressed delta store.

Only the delta is written, never the model.  Files are named by the SHA-256 of
their own bytes, so identical edits deduplicate and every manifest can be
verified against the file it describes.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from typing import Any, Dict, List

import numpy as np


class DeltaStore:
    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)

    def _path(self, sha: str) -> str:
        return os.path.join(self.root, sha[:2], sha + ".npy")

    def _mpath(self, sha: str) -> str:
        return os.path.join(self.root, sha[:2], sha + ".json")

    def put(self, delta, manifest: Dict[str, Any]) -> str:
        buf = io.BytesIO()
        np.save(buf, np.asarray(delta, dtype=np.float64))
        raw = buf.getvalue()
        sha = hashlib.sha256(raw).hexdigest()
        path = self._path(sha)
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(raw)
            record = dict(manifest)
            record["sha256"] = sha
            record["shape"] = list(np.shape(delta))
            record["bytes"] = len(raw)
            with open(self._mpath(sha), "w") as f:
                json.dump(record, f, indent=2, sort_keys=True)
        return sha

    def get(self, sha: str) -> np.ndarray:
        return np.load(self._path(sha))

    def manifest(self, sha: str) -> Dict[str, Any]:
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
        return os.path.exists(self._path(sha))
