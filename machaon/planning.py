"""Evidence-driven planning for a weight edit.

WSP is not a licence to edit whenever a number can be moved.  This module turns
diagnostic evidence into one of three explicit routes:

``edit``
    A local obstruction and a concrete site are both present.
``diagnose``
    The failure may be local, but the site evidence is missing.
``do_not_edit``
    The evidence describes a capability gap, not a removable obstruction.

Layer scores are supplied by the caller, usually from causal patching
(``machaon.podaliarius.patch_scores``) or another evaluation harness.  The
planner never guesses a layer and never writes weights.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

__all__ = [
    "FAILURE_TYPES",
    "PLAN_ROUTES",
    "EditPlan",
    "DiagnosisProtocol",
    "FailureDiagnosis",
    "diagnose_failure",
    "diagnose_and_plan",
    "scores_from_patch_report",
    "plan_surgery",
    "search_layers",
]

FAILURE_TYPES = ("obstruction", "capability_gap", "unknown")
PLAN_ROUTES = ("edit", "diagnose", "do_not_edit")


@dataclass(frozen=True)
class EditPlan:
    route: str
    failure_type: str
    reason: str
    layer: Optional[int] = None
    module_suffix: Optional[str] = None
    max_rel_norm: Optional[float] = None
    ranked_layers: Tuple[Tuple[int, float], ...] = ()
    next_probe: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "route": self.route,
            "failure_type": self.failure_type,
            "reason": self.reason,
            "layer": self.layer,
            "module_suffix": self.module_suffix,
            "max_rel_norm": self.max_rel_norm,
            "ranked_layers": [
                {"layer": int(layer), "score": float(score)}
                for layer, score in self.ranked_layers
            ],
            "next_probe": self.next_probe,
            "metadata": dict(self.metadata),
        }
        return payload


@dataclass(frozen=True)
class DiagnosisProtocol:
    """Thresholds for conservative failure classification."""

    target_metric: str = "accuracy"
    min_target_gap: float = 0.05
    min_mechanism_gap: float = 0.05
    mechanism_metrics: Tuple[str, ...] = ("loop_rate", "truncation_rate")

    def __post_init__(self) -> None:
        if not isinstance(self.target_metric, str) or not self.target_metric.strip():
            raise ValueError("target_metric must be a non-empty string")
        for name in ("min_target_gap", "min_mechanism_gap"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("%s must be finite and non-negative" % name)
        metrics = tuple(str(name) for name in self.mechanism_metrics)
        if not metrics or any(not name.strip() for name in metrics):
            raise ValueError("mechanism_metrics must be a non-empty tuple")
        object.__setattr__(self, "mechanism_metrics", metrics)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_metric": self.target_metric,
            "min_target_gap": float(self.min_target_gap),
            "min_mechanism_gap": float(self.min_mechanism_gap),
            "mechanism_metrics": list(self.mechanism_metrics),
        }


@dataclass(frozen=True)
class FailureDiagnosis:
    failure_type: str
    confidence: str
    reason: str
    target_metric: str
    target_gap: Optional[float]
    mechanism_improvements: Mapping[str, float] = field(default_factory=dict)
    next_step: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "failure_type": self.failure_type,
            "confidence": self.confidence,
            "reason": self.reason,
            "target_metric": self.target_metric,
            "target_gap": self.target_gap,
            "mechanism_improvements": dict(self.mechanism_improvements),
            "next_step": self.next_step,
            "metadata": dict(self.metadata),
        }


def _score_mapping(scores: Any) -> Dict[int, float]:
    if scores is None:
        return {}
    if isinstance(scores, Mapping):
        items = scores.items()
    elif isinstance(scores, Sequence) and not isinstance(scores, (str, bytes)):
        items = []
        for item in scores:
            if not isinstance(item, Mapping) or "layer" not in item or "score" not in item:
                raise ValueError("layer score rows need 'layer' and 'score'")
            items.append((item["layer"], item["score"]))
    else:
        raise ValueError("layer_scores must be a mapping or a sequence of rows")

    out: Dict[int, float] = {}
    for raw_layer, raw_score in items:
        try:
            layer = int(raw_layer)
            score = float(raw_score)
        except (TypeError, ValueError) as exc:
            raise ValueError("layer and score must be numeric") from exc
        if layer < 0:
            raise ValueError("layer must be non-negative")
        if not math.isfinite(score):
            raise ValueError("layer score must be finite")
        out[layer] = score
    return out


def scores_from_patch_report(
    report: Mapping[str, Any],
    key: str = "recovery",
) -> Dict[int, float]:
    """Extract layer scores from ``podaliarius.patch_scores`` output."""
    if not isinstance(report, Mapping):
        raise ValueError("patch report must be an object")
    scores = report.get("scores")
    if not isinstance(scores, Mapping):
        raise ValueError("patch report needs a 'scores' object")
    out = {}
    for layer, row in scores.items():
        if not isinstance(row, Mapping) or key not in row:
            raise ValueError("score row %r needs %r" % (layer, key))
        out[int(layer)] = float(row[key])
    return _score_mapping(out)


def _metric(state: Mapping[str, Any], name: str) -> float:
    if name not in state:
        raise KeyError("missing metric %r" % name)
    value = float(state[name])
    if not math.isfinite(value):
        raise ValueError("metric %r is not finite" % name)
    return value


def diagnose_failure(
    base: Mapping[str, Any],
    headroom: Optional[Mapping[str, Any]] = None,
    mechanism: Optional[Mapping[str, Any]] = None,
    protocol: Optional[DiagnosisProtocol] = None,
) -> FailureDiagnosis:
    """Classify the evidence without executing a model or editing weights.

    An obstruction requires both a target headroom gap and a mechanism
    improvement. A low headroom gap with no mechanism improvement is reported
    as a capability gap. Anything else remains unknown.
    """
    protocol = protocol or DiagnosisProtocol()
    if not isinstance(base, Mapping):
        raise ValueError("base must be an object")
    if headroom is not None and not isinstance(headroom, Mapping):
        raise ValueError("headroom must be an object")
    if headroom is None:
        return FailureDiagnosis(
            failure_type="unknown",
            confidence="low",
            reason="no headroom baseline is available",
            target_metric=protocol.target_metric,
            target_gap=None,
            next_step="collect a stronger unedited headroom baseline",
        )

    base_value = _metric(base, protocol.target_metric)
    headroom_value = _metric(headroom, protocol.target_metric)
    target_gap = headroom_value - base_value
    improvements: Dict[str, float] = {}
    if mechanism is not None:
        if not isinstance(mechanism, Mapping):
            raise ValueError("mechanism must be an object")
        mech_base = mechanism.get("base")
        mech_headroom = mechanism.get("headroom")
        if isinstance(mech_base, Mapping) and isinstance(mech_headroom, Mapping):
            for name in protocol.mechanism_metrics:
                if name in mech_base and name in mech_headroom:
                    improvements[name] = float(mech_base[name]) - float(mech_headroom[name])

    mechanism_relief = any(
        value >= protocol.min_mechanism_gap for value in improvements.values()
    )
    if target_gap < 0.0:
        return FailureDiagnosis(
            failure_type="unknown",
            confidence="low",
            reason="the purported headroom baseline scores below the base",
            target_metric=protocol.target_metric,
            target_gap=float(target_gap),
            mechanism_improvements=improvements,
            next_step="verify the headroom configuration and rerun the comparison",
        )
    if target_gap < protocol.min_target_gap:
        if mechanism_relief:
            return FailureDiagnosis(
                failure_type="unknown",
                confidence="low",
                reason="mechanism metrics improve but the target headroom gap is small",
                target_metric=protocol.target_metric,
                target_gap=float(target_gap),
                mechanism_improvements=improvements,
                next_step="collect a stronger target baseline before planning an edit",
            )
        return FailureDiagnosis(
            failure_type="capability_gap",
            confidence="medium" if improvements else "low",
            reason="the stronger decoding baseline does not improve the target and no mechanism relief is evidenced",
            target_metric=protocol.target_metric,
            target_gap=float(target_gap),
            mechanism_improvements=improvements,
            next_step="prefer data, training, retrieval or a broader intervention",
        )
    if not mechanism_relief:
        return FailureDiagnosis(
            failure_type="unknown",
            confidence="low",
            reason="the target headroom is higher, but mechanism relief is not evidenced",
            target_metric=protocol.target_metric,
            target_gap=float(target_gap),
            mechanism_improvements=improvements,
            next_step="collect loop/truncation evidence for both base and headroom",
        )
    all_relieved = bool(improvements) and all(
        value >= protocol.min_mechanism_gap for value in improvements.values()
    )
    return FailureDiagnosis(
        failure_type="obstruction",
        confidence="high" if all_relieved else "medium",
        reason="the target headroom and at least one mechanism metric improve together",
        target_metric=protocol.target_metric,
        target_gap=float(target_gap),
        mechanism_improvements=improvements,
        next_step="use plan_surgery with layer-site evidence",
    )


def diagnose_and_plan(
    base: Mapping[str, Any],
    headroom: Optional[Mapping[str, Any]] = None,
    mechanism: Optional[Mapping[str, Any]] = None,
    protocol: Optional[DiagnosisProtocol] = None,
    plan_kwargs: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a diagnosis and the corresponding fail-closed action plan."""
    diagnosis = diagnose_failure(base, headroom, mechanism, protocol)
    plan = plan_surgery(
        diagnosis.failure_type,
        **dict(plan_kwargs or {}),
    )
    return {"diagnosis": diagnosis.to_dict(), "plan": plan.to_dict()}


def plan_surgery(
    failure_type: str,
    *,
    layer_scores: Any = None,
    patch_report: Optional[Mapping[str, Any]] = None,
    module_suffix: str = "mlp.down_proj",
    max_rel_norm: float = 0.02,
    min_score: float = 0.0,
    metadata: Optional[Mapping[str, Any]] = None,
) -> EditPlan:
    """Choose an edit, a diagnosis, or a non-weight route from local evidence."""
    if failure_type not in FAILURE_TYPES:
        raise ValueError(
            "failure_type must be one of %s" % (", ".join(FAILURE_TYPES))
        )
    if not isinstance(module_suffix, str) or not module_suffix.strip():
        raise ValueError("module_suffix must be a non-empty string")
    if not (0.0 < float(max_rel_norm) <= 0.5):
        raise ValueError("max_rel_norm must be in (0, 0.5]")
    if not math.isfinite(float(min_score)):
        raise ValueError("min_score must be finite")
    if layer_scores is not None and patch_report is not None:
        raise ValueError("pass layer_scores or patch_report, not both")
    if patch_report is not None:
        layer_scores = scores_from_patch_report(patch_report)
    scores = _score_mapping(layer_scores)
    ranked = tuple(sorted(scores.items(), key=lambda item: (-item[1], item[0])))
    meta = dict(metadata or {})

    if failure_type == "capability_gap":
        return EditPlan(
            route="do_not_edit",
            failure_type=failure_type,
            reason="no local obstruction is evidenced; a weight patch is not the default route",
            ranked_layers=ranked,
            next_probe="collect capability/headroom evidence, or choose training, retrieval or a broader intervention",
            metadata=meta,
        )
    if failure_type == "unknown":
        return EditPlan(
            route="diagnose",
            failure_type=failure_type,
            reason="the failure has not been localized",
            ranked_layers=ranked,
            next_probe="run causal patching or mechanism metrics before selecting an edit site",
            metadata=meta,
        )
    if not ranked or ranked[0][1] <= float(min_score):
        return EditPlan(
            route="diagnose",
            failure_type=failure_type,
            reason="the obstruction is plausible but no layer clears the site-score threshold",
            ranked_layers=ranked,
            next_probe="widen the probe set or run causal patching across candidate layers",
            metadata=meta,
        )

    layer, score = ranked[0]
    meta["best_score"] = float(score)
    return EditPlan(
        route="edit",
        failure_type=failure_type,
        reason="a local obstruction and a positive site score are both present",
        layer=int(layer),
        module_suffix=module_suffix,
        max_rel_norm=float(max_rel_norm),
        ranked_layers=ranked,
        next_probe="propose a constrained edit at the selected layer, then require target, mechanism and control evidence",
        metadata=meta,
    )


def search_layers(
    layers: Sequence[int],
    evaluate: Callable[[int], Mapping[str, Any]],
    *,
    metric: str = "accuracy",
    higher_is_better: bool = True,
    max_evaluations: Optional[int] = None,
    cleanup: Optional[Callable[[], Any]] = None,
) -> Dict[str, Any]:
    """Evaluate candidate layers and rank successful trials.

    ``evaluate`` is responsible for constructing a proposal, applying it and
    collecting metrics. ``cleanup`` runs after every trial, including failures;
    a cleanup exception aborts the search instead of continuing from a dirty
    state. The helper itself is model-agnostic.
    """
    if not isinstance(metric, str) or not metric.strip():
        raise ValueError("metric must be a non-empty string")
    candidates = [int(layer) for layer in layers]
    if not candidates:
        raise ValueError("search_layers needs at least one candidate layer")
    if any(layer < 0 for layer in candidates):
        raise ValueError("candidate layers must be non-negative")
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidate layers must be unique")
    if max_evaluations is not None:
        max_evaluations = int(max_evaluations)
        if max_evaluations < 1:
            raise ValueError("max_evaluations must be positive")
        candidates = candidates[:max_evaluations]

    trials = []
    successes = []
    for layer in candidates:
        try:
            raw = evaluate(layer)
            if not isinstance(raw, Mapping):
                raise ValueError("evaluate(layer) must return a metrics object")
            metrics = dict(raw)
            if metric not in metrics:
                raise KeyError("missing ranking metric %r" % metric)
            score = float(metrics[metric])
            if not math.isfinite(score):
                raise ValueError("ranking metric %r is not finite" % metric)
            trial = {
                "layer": layer,
                "status": "ok",
                "score": score,
                "metrics": metrics,
                "error": None,
            }
            successes.append((layer, score))
        except Exception as exc:  # keep a per-layer failure record
            trial = {
                "layer": layer,
                "status": "error",
                "score": None,
                "metrics": {},
                "error": "%s: %s" % (type(exc).__name__, str(exc)),
            }
        finally:
            if cleanup is not None:
                cleanup()
        trials.append(trial)

    ranked = sorted(
        successes,
        key=lambda item: ((-item[1]) if higher_is_better else item[1], item[0]),
    )
    return {
        "metric": metric,
        "higher_is_better": bool(higher_is_better),
        "passed": bool(ranked),
        "best_layer": ranked[0][0] if ranked else None,
        "ranking": [
            {"layer": layer, "score": float(score)}
            for layer, score in ranked
        ],
        "trials": trials,
    }
