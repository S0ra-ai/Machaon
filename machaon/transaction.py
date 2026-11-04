"""Transactional weight application.

Weights do not get overwritten in place.  A transaction records the original
bytes, applies deltas, and either commits behind a verdict or restores the
workspace exactly.  Restoration is bit-exact because the originals are copied.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class Workspace:
    """A named set of weight matrices that can be edited atomically."""

    def __init__(self) -> None:
        self._w: Dict[str, np.ndarray] = {}
        self.history: List[Dict[str, Any]] = []

    def register(self, name: str, weight) -> None:
        arr = np.array(weight, dtype=np.float64, copy=True)
        if arr.ndim != 2:
            raise ValueError("weight %r must be 2-D, got %r" % (name, arr.shape))
        self._w[name] = arr

    def weight(self, name: str) -> np.ndarray:
        try:
            return self._w[name]
        except KeyError:
            raise KeyError(
                "unknown weight %r; known: %s" % (name, sorted(self._w))
            ) from None

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(sorted(self._w))

    def begin(self, label: str = "") -> "Transaction":
        return Transaction(self, label)


class Transaction:
    def __init__(self, workspace: Workspace, label: str = "") -> None:
        self._ws = workspace
        self.label = label
        self.deltas: List[Tuple[str, np.ndarray]] = []
        self._undo: List[Tuple[str, np.ndarray]] = []
        self._closed = False

    def apply(self, name: str, delta) -> np.ndarray:
        if self._closed:
            raise RuntimeError("transaction is already closed")
        original = self._ws.weight(name)
        delta = np.asarray(delta, dtype=np.float64)
        if delta.shape != original.shape:
            raise ValueError(
                "delta shape %r does not match weight %r shape %r"
                % (delta.shape, name, original.shape)
            )
        self._ws._w[name] = original + delta
        self._undo.append((name, original.copy()))
        self.deltas.append((name, delta))
        return self._ws._w[name]

    def rollback(self) -> None:
        for name, original in reversed(self._undo):
            self._ws._w[name] = original
        self._undo.clear()
        self._closed = True

    def commit(self, verdict: Optional[Any] = None) -> bool:
        """Commit, unless a verdict says otherwise.

        ``verdict`` is anything with a boolean ``passed`` attribute (see
        ``machaon.hygieia.gates.Verdict``).  A failed or missing verdict rolls
        back -- the default is fail-closed.
        """
        if self._closed:
            raise RuntimeError("transaction is already closed")
        if verdict is not None and not bool(getattr(verdict, "passed", False)):
            self.rollback()
            return False
        self._ws.history.append({"label": self.label, "deltas": list(self.deltas)})
        self._undo.clear()
        self._closed = True
        return True
