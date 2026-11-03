"""Checkpoint adapters.

The core is array-in / array-out.  Adapters are the only place that touches a
serialised model format.
"""
from __future__ import annotations

from .safetensors_store import Checkpoint, TensorInfo

__all__ = ["Checkpoint", "TensorInfo"]
