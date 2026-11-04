"""Edit procedures: how a spec becomes a delta."""
from __future__ import annotations

from .rome import RomeProposal, RomeResult, rome_edit, solve_edit

__all__ = ["RomeProposal", "RomeResult", "rome_edit", "solve_edit"]
