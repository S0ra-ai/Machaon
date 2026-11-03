"""Asclepius — the physician who decides *what* the new value should be.

Two gradient-free ways to obtain ``v``:
  * transplant it from a donor run that already knows the fact;
  * refine it with a (1+1) evolution strategy against a black-box score.
"""
from __future__ import annotations

from .solve import refine_value, value_from_donor

__all__ = ["value_from_donor", "refine_value"]
