"""The edit DSL.

An edit is a declarative record, not a Python call.  That is what makes it
auditable, replayable and safe to hand to an agent: the intent, the target,
the solver and the budget all travel together.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

OPS = {
    "rank1": "map one key to one value (ROME)",
    "rank1_constrained": "rank-1 edit leaving a protected key set untouched",
    "bake_direction": "bake a steering direction into a weight matrix",
    "ablate_output": "remove an output-space direction",
    "ablate_input": "remove an input-space direction",
    "low_rank_scale": "truncate and rescale a delta to rank r",
}

SOLVERS = {"closed_form", "donor", "search"}


@dataclass
class Target:
    module: str
    layer: Optional[int] = None

    def validate(self) -> None:
        if not isinstance(self.module, str) or not self.module.strip():
            raise ValueError("target.module must be a non-empty string")
        if self.layer is not None:
            if not isinstance(self.layer, int) or self.layer < 0:
                raise ValueError("target.layer must be a non-negative int or None")


@dataclass
class Constraints:
    max_rel_norm: float = 0.02
    preserve: str = "auto"

    def validate(self) -> None:
        if not (0.0 < float(self.max_rel_norm) <= 0.5):
            raise ValueError("constraints.max_rel_norm must be in (0, 0.5]")
        if not isinstance(self.preserve, str) or not self.preserve.strip():
            raise ValueError("constraints.preserve must be 'auto' or a key-set reference")


@dataclass
class EditSpec:
    op: str
    target: Target
    intent: str
    rank: int = 1
    solver: str = "closed_form"
    constraints: Constraints = field(default_factory=Constraints)

    def validate(self) -> "EditSpec":
        if self.op not in OPS:
            raise ValueError("unknown op %r; expected one of %s" % (self.op, sorted(OPS)))
        if self.solver not in SOLVERS:
            raise ValueError("unknown solver %r; expected one of %s" % (self.solver, sorted(SOLVERS)))
        if not isinstance(self.intent, str) or not self.intent.strip():
            raise ValueError("intent is required: it is the audit record of why this edit exists")
        if not isinstance(self.rank, int) or self.rank < 1:
            raise ValueError("rank must be a positive int")
        self.target.validate()
        self.constraints.validate()
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EditSpec":
        if "op" not in d or "target" not in d:
            raise ValueError("edit spec needs at least 'op' and 'target'")
        target = d["target"] if isinstance(d["target"], Target) else Target(**d["target"])
        cons_raw = d.get("constraints") or {}
        cons = cons_raw if isinstance(cons_raw, Constraints) else Constraints(**cons_raw)
        return cls(
            op=d["op"],
            target=target,
            intent=d.get("intent", ""),
            rank=int(d.get("rank", 1)),
            solver=d.get("solver", "closed_form"),
            constraints=cons,
        )
