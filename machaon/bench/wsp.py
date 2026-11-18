"""Executable WSP benchmark harness.

The benchmark does not ship a claim about model capability.  It ships a stable
schema for paired cases labelled ``obstruction`` or ``capability_gap`` and a
runner that compares the WSP decision with the expected repairability.  Real
benchmarks supply their own measured reports; the example fixture is synthetic
and exists to exercise the contract.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..wsp import WSPProtocol, assess_wsp

FORMAT = "machaon.wsp.benchmark"
FORMAT_VERSION = 1
FAILURE_TYPES = ("obstruction", "capability_gap")
SOURCE_KINDS = ("synthetic", "measured")

__all__ = [
    "FORMAT",
    "FORMAT_VERSION",
    "FAILURE_TYPES",
    "SOURCE_KINDS",
    "WSPBenchmarkCase",
    "evaluate_case",
    "run_benchmark",
    "load_benchmark",
]


@dataclass(frozen=True)
class WSPBenchmarkCase:
    case_id: str
    failure_type: str
    expected_repairable: bool
    base: Mapping[str, Any]
    candidate: Mapping[str, Any]
    controls: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    headroom: Optional[Mapping[str, Any]] = None
    mechanism: Optional[Mapping[str, Any]] = None
    gates: Optional[Mapping[str, Any]] = None
    protocol: Optional[Mapping[str, Any]] = None
    source_kind: str = "synthetic"
    provenance: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> "WSPBenchmarkCase":
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ValueError("case_id must be a non-empty string")
        if self.failure_type not in FAILURE_TYPES:
            raise ValueError(
                "failure_type must be one of %s" % (", ".join(FAILURE_TYPES))
            )
        if not isinstance(self.expected_repairable, bool):
            raise ValueError("expected_repairable must be a boolean")
        for name in ("base", "candidate"):
            if not isinstance(getattr(self, name), Mapping):
                raise ValueError("%s must be an object" % name)
        if not isinstance(self.controls, Mapping):
            raise ValueError("controls must be an object")
        if self.headroom is not None and not isinstance(self.headroom, Mapping):
            raise ValueError("headroom must be an object")
        if self.mechanism is not None and not isinstance(self.mechanism, Mapping):
            raise ValueError("mechanism must be an object")
        if self.gates is not None and not isinstance(self.gates, Mapping):
            raise ValueError("gates must be an object")
        if self.protocol is not None and not isinstance(self.protocol, Mapping):
            raise ValueError("protocol must be an object")
        if self.source_kind not in SOURCE_KINDS:
            raise ValueError(
                "source_kind must be one of %s" % (", ".join(SOURCE_KINDS))
            )
        if not isinstance(self.provenance, Mapping):
            raise ValueError("provenance must be an object")
        if self.source_kind == "measured":
            required = {
                "model", "checkpoint_sha256", "dataset", "dataset_sha256",
                "evaluated_at", "seed", "evaluation_script",
            }
            missing = sorted(name for name in required if name not in self.provenance)
            if missing:
                raise ValueError(
                    "measured provenance is missing: %s" % ", ".join(missing)
                )
            for name in ("checkpoint_sha256", "dataset_sha256"):
                digest = str(self.provenance[name])
                if not re.fullmatch(r"[0-9a-f]{64}", digest):
                    raise ValueError("%s must be a lowercase SHA-256" % name)
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be an object")
        return self

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        payload: Dict[str, Any] = {
            "case_id": self.case_id,
            "failure_type": self.failure_type,
            "expected_repairable": self.expected_repairable,
            "base": dict(self.base),
            "candidate": dict(self.candidate),
            "controls": {str(k): dict(v) for k, v in self.controls.items()},
            "source_kind": self.source_kind,
            "provenance": dict(self.provenance),
        }
        for name in ("headroom", "mechanism", "gates", "protocol", "metadata"):
            value = getattr(self, name)
            if value is not None:
                payload[name] = dict(value)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WSPBenchmarkCase":
        if not isinstance(payload, Mapping):
            raise ValueError("benchmark case must be an object")
        case = cls(
            case_id=payload.get("case_id", ""),
            failure_type=payload.get("failure_type", ""),
            expected_repairable=payload.get("expected_repairable"),
            base=payload.get("base", {}),
            candidate=payload.get("candidate", {}),
            controls=payload.get("controls", {}),
            headroom=payload.get("headroom"),
            mechanism=payload.get("mechanism"),
            gates=payload.get("gates"),
            protocol=payload.get("protocol"),
            source_kind=payload.get("source_kind", "synthetic"),
            provenance=payload.get("provenance", {}),
            metadata=payload.get("metadata", {}),
        )
        return case.validate()


def evaluate_case(case: WSPBenchmarkCase) -> Dict[str, Any]:
    """Run one case and compare its WSP decision with the expected outcome."""
    case = case.validate()
    protocol = WSPProtocol(**dict(case.protocol or {}))
    report = assess_wsp(
        case.base,
        case.candidate,
        controls=case.controls,
        headroom=case.headroom,
        gates=case.gates,
        protocol=protocol,
        mechanism=case.mechanism,
    )
    observed_repairable = bool(report["accepted"])
    return {
        "case_id": case.case_id,
        "failure_type": case.failure_type,
        "expected_repairable": case.expected_repairable,
        "observed_repairable": observed_repairable,
        "matched": observed_repairable == case.expected_repairable,
        "label": report["label"],
        "reason": report["reason"],
        "mechanism": report.get("mechanism"),
        "source_kind": case.source_kind,
        "provenance": dict(case.provenance),
        "metadata": dict(case.metadata),
        "report": report,
    }


def _summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    tp = sum(1 for row in rows
             if row["expected_repairable"] and row["observed_repairable"])
    fp = sum(1 for row in rows
             if not row["expected_repairable"] and row["observed_repairable"])
    tn = sum(1 for row in rows
             if not row["expected_repairable"] and not row["observed_repairable"])
    fn = sum(1 for row in rows
             if row["expected_repairable"] and not row["observed_repairable"])
    by_family: Dict[str, Dict[str, int]] = {}
    for row in rows:
        bucket = by_family.setdefault(
            row["failure_type"],
            {"n": 0, "expected_repairable": 0, "observed_repairable": 0, "matched": 0},
        )
        bucket["n"] += 1
        bucket["expected_repairable"] += int(row["expected_repairable"])
        bucket["observed_repairable"] += int(row["observed_repairable"])
        bucket["matched"] += int(row["matched"])
    source_kinds: Dict[str, int] = {}
    for row in rows:
        source_kinds[row["source_kind"]] = source_kinds.get(row["source_kind"], 0) + 1
    return {
        "n_cases": len(rows),
        "agreement": float(tp + tn) / len(rows) if rows else 0.0,
        "confusion": {
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn,
        },
        "by_family": by_family,
        "source_kinds": source_kinds,
        "contains_only_synthetic": set(source_kinds) <= {"synthetic"},
        "all_matched": tp + tn == len(rows),
    }


def run_benchmark(cases: Sequence[WSPBenchmarkCase]) -> Dict[str, Any]:
    """Run every case in deterministic order and return a portable report."""
    if not cases:
        raise ValueError("benchmark needs at least one case")
    ids = [case.case_id for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("case_id values must be unique")
    rows = [evaluate_case(case) for case in sorted(cases, key=lambda item: item.case_id)]
    return {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "summary": _summary(rows),
        "cases": rows,
    }


def load_benchmark(path: str | Path) -> List[WSPBenchmarkCase]:
    """Load cases from a JSON object with a ``cases`` array."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("benchmark file must contain an object")
    if payload.get("format") not in (None, FORMAT):
        raise ValueError("unsupported benchmark format: %r" % payload.get("format"))
    if payload.get("format_version") not in (None, FORMAT_VERSION):
        raise ValueError(
            "unsupported benchmark version: %r" % payload.get("format_version")
        )
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("benchmark file needs a non-empty 'cases' array")
    return [WSPBenchmarkCase.from_dict(item) for item in raw_cases]
