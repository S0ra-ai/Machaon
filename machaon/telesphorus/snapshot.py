"""Snapshot a workspace to disk so any edit can be undone later."""
from __future__ import annotations

from typing import Dict

import numpy as np

from ..transaction import Workspace


def save(workspace: Workspace, path: str) -> str:
    payload = {name: workspace.weight(name) for name in workspace.names}
    np.savez_compressed(path, **payload)
    return path


def load(path: str) -> Workspace:
    ws = Workspace()
    with np.load(path) as archive:
        for name in archive.files:
            ws.register(name, archive[name])
    return ws


def diff(before: Workspace, after: Workspace) -> Dict[str, Dict[str, object]]:
    """Relative change per weight, plus any shape mismatches."""
    out: Dict[str, Dict[str, object]] = {}
    names = sorted(set(before.names) | set(after.names))
    for name in names:
        if name not in before.names or name not in after.names:
            out[name] = {"present_in": "before" if name in before.names else "after"}
            continue
        a = before.weight(name)
        b = after.weight(name)
        if a.shape != b.shape:
            out[name] = {"shape_mismatch": [list(a.shape), list(b.shape)]}
            continue
        denom = float(np.linalg.norm(a))
        delta = float(np.linalg.norm(b - a))
        out[name] = {
            "rel_change": (delta / denom) if denom else float("inf"),
            "changed": bool(delta > 0.0),
        }
    return out
