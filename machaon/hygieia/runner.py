"""Run an edit through the gates.

The runner is backend-agnostic: it needs ``next_logits(items)`` and ``nll(items)``
and nothing else.  That is what lets the same suite run against a 40k-parameter
toy and an 8B checkpoint without changing a line of metrics code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .gates import Verdict, evaluate
from .metrics import kl_summary, perplexity_ratio, softmax


@dataclass
class EditSuite:
    """What to measure.  Items are whatever the backend accepts."""

    target_items: Sequence[Any]
    target_token_ids: Sequence[int]
    neutral_items: Sequence[Any] = ()
    corpus_items: Sequence[Any] = ()
    extra: Dict[str, float] = field(default_factory=dict)


def _first_token_accuracy(logits: np.ndarray, expected: Sequence[int]) -> float:
    expected = np.asarray(list(expected), dtype=np.int64)
    if logits.shape[0] != expected.size:
        raise ValueError(
            "got %d rows of logits for %d targets" % (logits.shape[0], expected.size)
        )
    pred = np.asarray(logits).argmax(axis=-1)
    return float((pred == expected).mean())


def evaluate_edit(before: Any, after: Any, suite: EditSuite) -> Dict[str, float]:
    """Measure an edit.  ``before`` and ``after`` are the same model at two states."""
    if not suite.target_items:
        raise ValueError("suite has no target items; nothing to measure")

    metrics: Dict[str, float] = {}

    # 1. did the edit land?
    after_target = after.next_logits(suite.target_items)
    metrics["rewrite_acc"] = _first_token_accuracy(after_target, suite.target_token_ids)

    if suite.neutral_items:
        # 2. did anything else move?
        kl = kl_summary(
            softmax(before.next_logits(suite.neutral_items)),
            softmax(after.next_logits(suite.neutral_items)),
        )
        metrics["kl_p50"] = kl["kl_p50"]
        metrics["kl_p95"] = kl["kl_p95"]
        metrics["kl_max"] = kl["kl_max"]

    if suite.corpus_items:
        # 3. is the model still a model?
        metrics["ppl_ratio"] = perplexity_ratio(
            before.nll(suite.corpus_items), after.nll(suite.corpus_items)
        )

    metrics.update(suite.extra)
    return metrics


def judge(before: Any, after: Any, suite: EditSuite, gates: Optional[Dict] = None) -> Verdict:
    """Convenience: measure, then apply the acceptance gates (fail-closed)."""
    return evaluate(evaluate_edit(before, after, suite), gates)
