"""Local web interface for Machaon.

The web package is deliberately optional: importing :mod:`machaon` does not
import FastAPI, uvicorn, torch or any other web dependency.  The console entry
point lives in :mod:`machaon.web.app`.
"""
from __future__ import annotations

__all__ = ["create_app", "main"]


def __getattr__(name):
    """Keep the optional FastAPI dependency out of the core import path."""
    if name in __all__:
        from . import app

        return getattr(app, name)
    raise AttributeError(name)
