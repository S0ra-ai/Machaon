"""WSP acceptance protocol.

The question is not only whether an edit improves a target metric.  It is
whether the improvement is *specific* to the edit and whether it removes an
obstruction or adds a capability that was not already present.

This module implements the acceptance protocol described in the WSP report:

* generate same-size, same-support random controls for a delta;
* compare a candidate against those controls;
* compare it against an optional "headroom" baseline (for example, the base
  model with thinking disabled or a larger decoding budget);
* label the result as ``repair``, ``acquisition``, ``non_specific``,
  ``no_effect``, ``regression`` or ``unclassified``;
* keep the acceptance decision fail-closed when the normal gates are missing.

Nothing here loads a model or executes an edit.  It is a pure decision layer
that can sit above :mod:`machaon.hygieia` and :mod:`machaon.session`.
"""
from __future__ import annotations

import math
import zlib
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .hygieia.gates import GATES, evaluate
from .mechanism import MechanismProtocol, assess_mechanism

__all__ = [
    "WSPProtocol",
    "EffectVerdict",
    "random_control",
    "random_controls",
    "classify_effect",
    "assess_wsp",
    "assess_report",
]


@dataclass(frozen=True)
class WSPProtocol:
    """Thresholds for one WSP acceptance decision.

    ``min_gain`` is the smallest target-metric increase worth calling an
    effect.  ``control_margin`` is how close a matched control may get to the
    candidate before the candidate is called non-specific.  ``headroom_margin``
    is how close an alternative baseline must be to the candidate before the
    effect is called a repair rather than an acquisition.
    """

    target_metric: str = "rewrite_acc"
    min_gain: float = 0.05
    control_margin: float = 0.05
    headroom_margin: float = 0.05
    require_gates: bool = True
    require_controls: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.target_metric, str) or not self.target_metric.strip():
            raise ValueError("target_metric must be a non-empty string")
        for name in ("min_gain", "control_margin", "headroom_margin"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("%s must be a finite non-negative number" % name)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EffectVerdict:
    label: str
    reason: str
    accepted: bool
    target_metric: str
    base_value: float
    candidate_value: float
    gain: float
    best_control_name: Optional[str] = None
    best_control_value: Optional[float] = None
    control_gap: Optional[float] = None
    headroom_name: Optional[str] = None
    headroom_value: Optional[float] = None
    headroom_gap: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _metric_value(state: Mapping[str, Any], metric: str) -> float:
    if metric not in state:
        raise KeyError("missing target metric %r" % metric)
    value = float(state[metric])
    if not math.isfinite(value):
        raise ValueError("metric %r is not finite" % metric)
    return value


def _control_items(
    controls: Optional[Mapping[str, Mapping[str, Any]]],
) -> Sequence[Tuple[str, Mapping[str, Any]]]:
    if not controls:
        return ()
    return tuple(sorted(controls.items()))


def random_control(
    delta: Any,
    seed: int = 0,
    preserve_support: bool = True,
) -> np.ndarray:
    """Return a deterministic random delta with the same shape and norm.

    With ``preserve_support=True`` (default), only entries that are nonzero in
    ``delta`` are randomized.  This is a closer control for sparse edits than a
    dense Gaussian matrix: the control has the same shape, support and
    Frobenius norm, but no claim to the candidate's direction.
    """
    array = np.asarray(delta, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("delta must be a 2-D array")
    if not np.isfinite(array).all():
        raise ValueError("delta contains non-finite values")
    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        return np.zeros_like(array, dtype=np.float64)

    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    support = array != 0.0 if preserve_support else np.ones(array.shape, dtype=bool)
    if not support.any():
        return np.zeros_like(array, dtype=np.float64)

    out = np.zeros_like(array, dtype=np.float64)
    out[support] = rng.standard_normal(int(support.sum()))
    out_norm = float(np.linalg.norm(out))
    if out_norm == 0.0:
        raise ValueError("random control collapsed to zero; choose another seed")
    return out * (norm / out_norm)


def random_controls(
    deltas: Mapping[str, Any],
    seed: int = 0,
    preserve_support: bool = True,
) -> Dict[str, np.ndarray]:
    """Build deterministic matched controls for a whole edit.

    Each tensor gets its own derived seed, so adding or removing a tensor does
    not change the controls for the others and the result is reproducible from
    the edit manifest alone.
    """
    controls: Dict[str, np.ndarray] = {}
    for name in sorted(deltas):
        tensor_seed = (int(seed) + zlib.crc32(name.encode("utf-8"))) & 0xFFFFFFFF
        controls[name] = random_control(
            deltas[name], seed=tensor_seed, preserve_support=preserve_support
        )
    return controls


def classify_effect(
    base: Mapping[str, Any],
    candidate: Mapping[str, Any],
    controls: Optional[Mapping[str, Mapping[str, Any]]] = None,
    headroom: Optional[Mapping[str, Any]] = None,
    protocol: Optional[WSPProtocol] = None,
) -> EffectVerdict:
    """Classify what an edit did, without assuming it added a capability.

    ``headroom`` is the score of a stronger baseline for the same target, such
    as the base model with a different decoding mode.  Without it, a positive
    result is deliberately labelled ``unclassified``: an improvement alone
    cannot distinguish "removed an obstruction" from "added a capability".
    """
    protocol = protocol or WSPProtocol()
    metric = protocol.target_metric
    base_value = _metric_value(base, metric)
    candidate_value = _metric_value(candidate, metric)
    gain = candidate_value - base_value

    control_items = _control_items(controls)
    best_name: Optional[str] = None
    best_value: Optional[float] = None
    for name, state in control_items:
        value = _metric_value(state, metric)
        if best_value is None or value > best_value:
            best_name, best_value = name, value
    control_gap = None if best_value is None else candidate_value - best_value

    headroom_name: Optional[str] = None
    headroom_value: Optional[float] = None
    if headroom is not None:
        headroom_name = str(headroom.get("name", "headroom"))
        headroom_value = _metric_value(headroom, metric)
    headroom_gap = None if headroom_value is None else candidate_value - headroom_value

    if gain < -protocol.min_gain:
        return EffectVerdict(
            label="regression",
            reason="candidate is worse than the base by %.4f" % (-gain),
            accepted=False,
            target_metric=metric,
            base_value=base_value,
            candidate_value=candidate_value,
            gain=gain,
            best_control_name=best_name,
            best_control_value=best_value,
            control_gap=control_gap,
            headroom_name=headroom_name,
            headroom_value=headroom_value,
            headroom_gap=headroom_gap,
        )
    if abs(gain) < protocol.min_gain:
        return EffectVerdict(
            label="no_effect",
            reason="target gain %.4f is below the %.4f threshold" % (gain, protocol.min_gain),
            accepted=False,
            target_metric=metric,
            base_value=base_value,
            candidate_value=candidate_value,
            gain=gain,
            best_control_name=best_name,
            best_control_value=best_value,
            control_gap=control_gap,
            headroom_name=headroom_name,
            headroom_value=headroom_value,
            headroom_gap=headroom_gap,
        )
    if protocol.require_controls and not control_items:
        return EffectVerdict(
            label="inconclusive",
            reason="a matched control is required before accepting the effect",
            accepted=False,
            target_metric=metric,
            base_value=base_value,
            candidate_value=candidate_value,
            gain=gain,
            headroom_name=headroom_name,
            headroom_value=headroom_value,
            headroom_gap=headroom_gap,
        )
    if best_value is not None and control_gap <= protocol.control_margin:
        return EffectVerdict(
            label="non_specific",
            reason="matched control %r reaches %.4f, within %.4f of the candidate"
            % (best_name, best_value, protocol.control_margin),
            accepted=False,
            target_metric=metric,
            base_value=base_value,
            candidate_value=candidate_value,
            gain=gain,
            best_control_name=best_name,
            best_control_value=best_value,
            control_gap=control_gap,
            headroom_name=headroom_name,
            headroom_value=headroom_value,
            headroom_gap=headroom_gap,
        )
    if headroom_value is None:
        return EffectVerdict(
            label="unclassified",
            reason="positive, control-specific gain; add a headroom baseline to distinguish repair from acquisition",
            accepted=False,
            target_metric=metric,
            base_value=base_value,
            candidate_value=candidate_value,
            gain=gain,
            best_control_name=best_name,
            best_control_value=best_value,
            control_gap=control_gap,
        )
    if headroom_gap <= protocol.headroom_margin:
        return EffectVerdict(
            label="repair",
            reason="candidate reaches a baseline that already had the capability (headroom gap %.4f)"
            % headroom_gap,
            accepted=True,
            target_metric=metric,
            base_value=base_value,
            candidate_value=candidate_value,
            gain=gain,
            best_control_name=best_name,
            best_control_value=best_value,
            control_gap=control_gap,
            headroom_name=headroom_name,
            headroom_value=headroom_value,
            headroom_gap=headroom_gap,
        )
    return EffectVerdict(
        label="acquisition",
        reason="candidate exceeds both the matched controls and the headroom baseline",
        accepted=True,
        target_metric=metric,
        base_value=base_value,
        candidate_value=candidate_value,
        gain=gain,
        best_control_name=best_name,
        best_control_value=best_value,
        control_gap=control_gap,
        headroom_name=headroom_name,
        headroom_value=headroom_value,
        headroom_gap=headroom_gap,
    )


def _gate_report(metrics: Mapping[str, Any], gates: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    gate_map = None
    if gates:
        gate_map = {str(k): (str(v[0]), float(v[1])) for k, v in gates.items()}
    verdict = evaluate(dict(metrics), gate_map or GATES)
    return {
        "passed": verdict.passed,
        "failures": list(verdict.failures),
        "missing": list(verdict.missing),
        "n_gates": verdict.n_gates,
    }


def assess_wsp(
    base: Mapping[str, Any],
    candidate: Mapping[str, Any],
    controls: Optional[Mapping[str, Mapping[str, Any]]] = None,
    headroom: Optional[Mapping[str, Any]] = None,
    gates: Optional[Mapping[str, Any]] = None,
    protocol: Optional[WSPProtocol] = None,
    mechanism: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the WSP decision layer and return a serialisable report."""
    protocol = protocol or WSPProtocol()
    verdict = classify_effect(base, candidate, controls, headroom, protocol)
    candidate_gate = _gate_report(candidate, gates)
    control_gates = {
        name: _gate_report(state, gates)
        for name, state in _control_items(controls)
    }
    mechanism_report = None
    if mechanism is not None:
        if not isinstance(mechanism, Mapping):
            raise ValueError("mechanism must be an object")
        raw_protocol = mechanism.get("protocol") or {}
        if not isinstance(raw_protocol, Mapping):
            raise ValueError("mechanism.protocol must be an object")
        mechanism_protocol = MechanismProtocol.from_dict(raw_protocol)
        for key in ("base", "candidate"):
            if key not in mechanism or not isinstance(mechanism[key], Mapping):
                raise ValueError("mechanism needs a %r metrics object" % key)
        mechanism_controls = mechanism.get("controls") or {}
        if not isinstance(mechanism_controls, Mapping):
            raise ValueError("mechanism.controls must be an object")
        mechanism_headroom = mechanism.get("headroom")
        if mechanism_headroom is not None and not isinstance(mechanism_headroom, Mapping):
            raise ValueError("mechanism.headroom must be an object")
        mechanism_report = assess_mechanism(
            mechanism["base"],
            mechanism["candidate"],
            controls=mechanism_controls,
            headroom=mechanism_headroom,
            protocol=mechanism_protocol,
        ).to_dict()
    accepted = bool(
        verdict.accepted
        and (not protocol.require_gates or candidate_gate["passed"])
        and (mechanism_report is None or mechanism_report["passed"])
    )
    if verdict.accepted and protocol.require_gates and not candidate_gate["passed"]:
        reason = "effect is %s, but candidate gates failed" % verdict.label
    elif verdict.accepted and mechanism_report is not None and not mechanism_report["passed"]:
        reason = "target effect is %s, but mechanism gate failed: %s" % (
            verdict.label, mechanism_report["reason"]
        )
    else:
        reason = verdict.reason
    report = {
        "accepted": accepted,
        "label": verdict.label,
        "reason": reason,
        "classification": verdict.to_dict(),
        "candidate_gate": candidate_gate,
        "control_gates": control_gates,
        "protocol": protocol.to_dict(),
    }
    if mechanism_report is not None:
        report["mechanism"] = mechanism_report
    return report


def assess_report(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Assess a JSON-shaped report.

    Accepted shape::

        {
          "base": {"rewrite_acc": 0.20, ...},
          "candidate": {"rewrite_acc": 0.90, ...},
          "controls": {"random": {"rewrite_acc": 0.20, ...}},
          "headroom": {"name": "thinking_off", "rewrite_acc": 0.86, ...},
          "mechanism": {                                  # optional
            "base": {"loop_rate": 0.90, ...},
            "candidate": {"loop_rate": 0.30, ...},
            "controls": {"random": {"loop_rate": 0.85, ...}},
            "protocol": {"metrics": ["loop_rate"], ...}
          },
          "gates": {"rewrite_acc": [">=", 0.95], ...},   # optional
          "protocol": {"target_metric": "rewrite_acc"}   # optional
        }
    """
    if not isinstance(payload, Mapping):
        raise ValueError("report must be an object")
    for key in ("base", "candidate"):
        if key not in payload or not isinstance(payload[key], Mapping):
            raise ValueError("report needs a %r metrics object" % key)
    controls = payload.get("controls") or {}
    if not isinstance(controls, Mapping):
        raise ValueError("controls must be an object mapping names to metrics")
    for name, state in controls.items():
        if not isinstance(state, Mapping):
            raise ValueError("control %r must be a metrics object" % name)
    headroom = payload.get("headroom")
    if headroom is not None and not isinstance(headroom, Mapping):
        raise ValueError("headroom must be a metrics object")
    gates = payload.get("gates")
    if gates is not None and not isinstance(gates, Mapping):
        raise ValueError("gates must be an object mapping metric names to [op, threshold]")
    raw_protocol = payload.get("protocol") or {}
    if not isinstance(raw_protocol, Mapping):
        raise ValueError("protocol must be an object")
    protocol = WSPProtocol(**dict(raw_protocol))
    mechanism = payload.get("mechanism")
    if mechanism is not None and not isinstance(mechanism, Mapping):
        raise ValueError("mechanism must be an object")
    return assess_wsp(
        payload["base"],
        payload["candidate"],
        controls=controls,
        headroom=headroom,
        gates=gates,
        protocol=protocol,
        mechanism=mechanism,
    )
