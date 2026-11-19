"""Report which compute devices the installed runtime can see.

Machaon loads models on CPU unless the caller asks otherwise. This module
answers the question a first-time user actually has - "can I use my GPU?" -
without loading a model or allocating memory.
"""
from __future__ import annotations

from typing import Any, Dict

__all__ = ["device_report"]


def device_report() -> Dict[str, Any]:
    """Availability of CPU, CUDA and Apple MPS for the installed torch."""
    report: Dict[str, Any] = {
        "default": "cpu",
        "torch": False,
        "cuda": False,
        "cuda_count": 0,
        "mps": False,
        "mps_built": False,
    }
    try:
        import torch
    except ImportError:
        return report
    report["torch"] = True
    try:
        report["cuda"] = bool(torch.cuda.is_available())
        report["cuda_count"] = int(torch.cuda.device_count())
    except (RuntimeError, AttributeError):
        pass
    backend = getattr(torch.backends, "mps", None)
    if backend is not None:
        try:
            report["mps"] = bool(backend.is_available())
            report["mps_built"] = bool(backend.is_built())
        except (RuntimeError, AttributeError):
            pass
    return report
