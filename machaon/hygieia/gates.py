"""Acceptance gates.

These thresholds are opinions, and they are deliberately strict.  Change them
per project, but change them *before* running the edit, not after seeing the
numbers.

Fail-closed rules:
  * a metric that is absent counts as a failure;
  * a verdict with no failures but missing metrics never passes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

GATES: Dict[str, Tuple[str, float]] = {
    # target achieved
    "rewrite_acc": (">=", 0.95),
    "paraphrase_acc": (">=", 0.60),
    # nothing else moved
    "unrelated_acc": (">=", 1.00),
    "kl_p95": ("<=", 0.05),
    "kl_max": ("<=", 0.50),
    # capability preserved
    "mmlu_delta": (">=", -0.01),
    "ppl_ratio": ("<=", 1.02),
    # no self-contradiction in long-form generation
    "longform_contradictions": ("==", 0.0),
}


@dataclass
class Verdict:
    passed: bool
    failures: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.passed

    def summary(self) -> str:
        if self.passed:
            return "PASS (%d gates)" % len(self.metrics)
        lines = ["FAIL"]
        lines += ["  missing: %s" % m for m in self.missing]
        lines += ["  %s" % f for f in self.failures]
        return "\n".join(lines)


def _cmp(value: float, op: str, threshold: float) -> bool:
    if op == ">=":
        return value >= threshold
    if op == "<=":
        return value <= threshold
    if op == "==":
        return value == threshold
    if op == ">":
        return value > threshold
    if op == "<":
        return value < threshold
    raise ValueError("unsupported gate operator %r" % (op,))


def _fmt(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else ("%g" % x)


def evaluate(metrics: Dict[str, float], gates: Dict[str, Tuple[str, float]] = None) -> Verdict:
    gates = GATES if gates is None else gates
    failures: List[str] = []
    missing: List[str] = []
    for name, (op, threshold) in gates.items():
        if name not in metrics:
            missing.append(name)
            continue
        value = float(metrics[name])
        if not _cmp(value, op, threshold):
            failures.append(
                "%s = %s violates %s %s" % (name, _fmt(value), op, _fmt(threshold))
            )
    passed = not failures and not missing
    return Verdict(passed=passed, failures=failures, missing=missing, metrics=dict(metrics))
