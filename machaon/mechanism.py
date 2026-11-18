"""Mechanism-level evidence for WSP decisions.

Target accuracy alone cannot tell an obstruction repair from a capability
addition.  These helpers turn generated traces into the mechanism metrics the
WSP report asks for: repeated-loop rate, empty-output rate and the fraction of
generations that hit the token limit.  ``assess_mechanism`` then checks whether
the candidate improves those metrics relative to the unedited base and whether
a matched control improves them too.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .hygieia.longform import repeated_ngram_rate

__all__ = [
    "MechanismProtocol",
    "MechanismVerdict",
    "generation_metrics",
    "records_metrics",
    "generation_mechanism",
    "assess_mechanism",
]

_DIRECTIONS = ("lower", "higher")


@dataclass(frozen=True)
class MechanismProtocol:
    """Directions and margins for one obstruction gate."""

    metrics: Tuple[str, ...] = ("loop_rate", "truncation_rate")
    directions: Tuple[Tuple[str, str], ...] = (
        ("loop_rate", "lower"),
        ("truncation_rate", "lower"),
    )
    min_improvement: float = 0.05
    control_margin: float = 0.05
    require_controls: bool = True

    def __post_init__(self) -> None:
        metrics = tuple(str(name) for name in self.metrics)
        if not metrics or any(not name.strip() for name in metrics):
            raise ValueError("metrics must be a non-empty tuple of names")
        directions = tuple((str(k), str(v)) for k, v in self.directions)
        configured = dict(directions)
        for name in metrics:
            if name not in configured:
                raise ValueError("metric %r needs a direction" % name)
            if configured[name] not in _DIRECTIONS:
                raise ValueError("direction for %r must be lower or higher" % name)
        for name in ("min_improvement", "control_margin"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("%s must be a finite non-negative number" % name)
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "directions", directions)

    def direction(self, metric: str) -> str:
        return dict(self.directions)[metric]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MechanismProtocol":
        directions = payload.get("directions")
        if isinstance(directions, Mapping):
            directions = tuple(directions.items())
        return cls(
            metrics=tuple(payload.get("metrics", cls.metrics)),
            directions=tuple(directions or cls.directions),
            min_improvement=float(payload.get("min_improvement", 0.05)),
            control_margin=float(payload.get("control_margin", 0.05)),
            require_controls=bool(payload.get("require_controls", True)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metrics": list(self.metrics),
            "directions": dict(self.directions),
            "min_improvement": float(self.min_improvement),
            "control_margin": float(self.control_margin),
            "require_controls": bool(self.require_controls),
        }


@dataclass(frozen=True)
class MechanismVerdict:
    passed: bool
    label: str
    reason: str
    metrics: Dict[str, Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "label": self.label,
            "reason": self.reason,
            "metrics": self.metrics,
        }


def generation_metrics(
    texts: Sequence[str],
    *,
    hit_token_limit: Optional[Sequence[bool]] = None,
    generated_tokens: Optional[Sequence[int]] = None,
    ngram: int = 4,
) -> Dict[str, Any]:
    """Compute mechanism metrics from generated text and optional trace facts."""
    texts = [str(text) for text in texts]
    if not texts:
        raise ValueError("generation_metrics needs at least one text")
    if int(ngram) < 1:
        raise ValueError("ngram must be positive")
    if hit_token_limit is not None and len(hit_token_limit) != len(texts):
        raise ValueError("hit_token_limit must match the number of texts")
    if generated_tokens is not None and len(generated_tokens) != len(texts):
        raise ValueError("generated_tokens must match the number of texts")

    report: Dict[str, Any] = {
        "n": len(texts),
        "loop_rate": float(np.mean([repeated_ngram_rate(text, ngram) for text in texts])),
        "empty_rate": float(np.mean([1.0 if not text.strip() else 0.0 for text in texts])),
    }
    if hit_token_limit is not None:
        report["truncation_rate"] = float(np.mean([1.0 if hit else 0.0 for hit in hit_token_limit]))
    if generated_tokens is not None:
        values = [int(value) for value in generated_tokens]
        report["mean_generated_tokens"] = float(np.mean(values))
        report["max_generated_tokens"] = int(max(values))
    return report


def records_metrics(records: Sequence[Any], *, ngram: int = 4) -> Dict[str, Any]:
    """Compute mechanism metrics from ``generate_records`` output.

    Strings are accepted as a convenience; they produce loop and empty rates
    but no truncation rate because the stop reason is unavailable.
    """
    if not records:
        raise ValueError("records_metrics needs at least one record")
    if all(isinstance(record, str) for record in records):
        return generation_metrics(records, ngram=ngram)
    texts = []
    limits = []
    token_counts = []
    have_limits = True
    have_tokens = True
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("records must be strings or mappings")
        texts.append(str(record.get("text", "")))
        if "hit_token_limit" in record:
            limits.append(bool(record["hit_token_limit"]))
        else:
            have_limits = False
        if "generated_tokens" in record:
            token_counts.append(int(record["generated_tokens"]))
        else:
            have_tokens = False
    return generation_metrics(
        texts,
        hit_token_limit=limits if have_limits else None,
        generated_tokens=token_counts if have_tokens else None,
        ngram=ngram,
    )


def generation_mechanism(
    base: Sequence[Any],
    candidate: Sequence[Any],
    controls: Optional[Mapping[str, Sequence[Any]]] = None,
    headroom: Optional[Sequence[Any]] = None,
    *,
    ngram: int = 4,
) -> Dict[str, Any]:
    """Build a serialisable mechanism-evidence payload from generation traces."""
    payload: Dict[str, Any] = {
        "base": records_metrics(base, ngram=ngram),
        "candidate": records_metrics(candidate, ngram=ngram),
        "controls": {
            str(name): records_metrics(records, ngram=ngram)
            for name, records in sorted((controls or {}).items())
        },
    }
    if headroom is not None:
        payload["headroom"] = records_metrics(headroom, ngram=ngram)
    return payload


def _value(state: Mapping[str, Any], metric: str) -> float:
    if metric not in state:
        raise KeyError("missing mechanism metric %r" % metric)
    value = float(state[metric])
    if not math.isfinite(value):
        raise ValueError("mechanism metric %r is not finite" % metric)
    return value


def _best_control(
    controls: Mapping[str, Mapping[str, Any]], metric: str, direction: str
) -> Tuple[Optional[str], Optional[float]]:
    name = None
    value = None
    for control_name, state in sorted(controls.items()):
        current = _value(state, metric)
        if value is None:
            name, value = control_name, current
        elif direction == "lower" and current < value:
            name, value = control_name, current
        elif direction == "higher" and current > value:
            name, value = control_name, current
    return name, value


def assess_mechanism(
    base: Mapping[str, Any],
    candidate: Mapping[str, Any],
    controls: Optional[Mapping[str, Mapping[str, Any]]] = None,
    headroom: Optional[Mapping[str, Any]] = None,
    protocol: Optional[MechanismProtocol] = None,
) -> MechanismVerdict:
    """Gate an obstruction repair on mechanism metrics.

    Every configured metric must improve by ``min_improvement`` from the base
    and beat the best matched control by ``control_margin``.  A control is
    required by default so "the model improved" cannot be called a mechanism
    repair without a specificity check.
    """
    protocol = protocol or MechanismProtocol()
    controls = dict(controls or {})
    rows: Dict[str, Dict[str, Any]] = {}
    missing = []
    non_specific = []
    no_effect = []

    for metric in protocol.metrics:
        try:
            base_value = _value(base, metric)
            candidate_value = _value(candidate, metric)
        except (KeyError, ValueError) as exc:
            missing.append(str(exc))
            continue
        direction = protocol.direction(metric)
        improvement = (
            base_value - candidate_value
            if direction == "lower"
            else candidate_value - base_value
        )
        row: Dict[str, Any] = {
            "base": base_value,
            "candidate": candidate_value,
            "direction": direction,
            "improvement": float(improvement),
            "passed": bool(improvement >= protocol.min_improvement),
        }
        if improvement < protocol.min_improvement:
            no_effect.append(metric)
        try:
            control_name, best_control = _best_control(controls, metric, direction)
        except (KeyError, ValueError) as exc:
            missing.append("%s control: %s" % (metric, exc))
            control_name, best_control = None, None
        row["best_control_name"] = control_name
        row["best_control"] = best_control
        if best_control is not None:
            gap = (
                best_control - candidate_value
                if direction == "lower"
                else candidate_value - best_control
            )
            row["control_gap"] = float(gap)
            if gap < protocol.control_margin:
                row["passed"] = False
                non_specific.append(metric)
        elif protocol.require_controls:
            missing.append("%s needs a matched control" % metric)
            row["passed"] = False
        if headroom is not None:
            try:
                headroom_value = _value(headroom, metric)
            except (KeyError, ValueError) as exc:
                missing.append("%s headroom: %s" % (metric, exc))
            else:
                row["headroom"] = headroom_value
                row["headroom_gap"] = float(
                    candidate_value - headroom_value
                    if direction == "lower"
                    else headroom_value - candidate_value
                )
        rows[metric] = row

    if missing:
        passed = False
        label = "inconclusive"
        reason = "; ".join(missing)
    elif non_specific:
        passed = False
        label = "non_specific"
        reason = "matched control reaches the candidate on %s" % ", ".join(non_specific)
    elif no_effect:
        passed = False
        label = "no_effect"
        reason = "mechanism improvement below %.4f on %s" % (
            protocol.min_improvement, ", ".join(no_effect)
        )
    else:
        passed = True
        label = "obstruction_relieved"
        reason = "candidate improves every configured mechanism metric over the matched control"
    return MechanismVerdict(
        passed=passed,
        label=label,
        reason=reason,
        metrics=rows,
    )
