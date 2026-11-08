"""Machaon — direct weight surgery for LLMs.

Extract the arrow. Leave the body intact.

The core is pure NumPy so the maths is testable anywhere; PyTorch and
Hugging Face are optional adapters.
"""
from __future__ import annotations

from .contract import ContractSolution, synthesize_contract
from .linalg import (
    ablate_input,
    ablate_output,
    bake_direction,
    clamp_rel_norm,
    low_rank_scale,
    nullspace_basis,
    nullspace_projector,
    project_complement,
    rank1,
    rank1_constrained,
    relative_norm,
)
from .schema import Constraints, EditSpec, Target
from .transaction import Transaction, Workspace

__version__ = "0.3.4"

__all__ = [
    "rank1",
    "rank1_constrained",
    "bake_direction",
    "ablate_output",
    "ablate_input",
    "low_rank_scale",
    "nullspace_basis",
    "nullspace_projector",
    "project_complement",
    "clamp_rel_norm",
    "relative_norm",
    "ContractSolution",
    "synthesize_contract",
    "EditSpec",
    "Target",
    "Constraints",
    "Workspace",
    "Transaction",
]
