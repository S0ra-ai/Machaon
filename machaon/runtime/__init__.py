"""Model runtimes.

A runtime is anything that can produce next-token logits and token-level NLL.
The evaluation harness depends only on that contract, so adding an architecture
means adding a runtime, not touching the metrics.
"""
from __future__ import annotations

from .hf import HFModel, layer_container

__all__ = ["HFModel", "layer_container"]
