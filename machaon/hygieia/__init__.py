"""Hygieia — prevention.

Nothing is committed until it passes the gates.  Missing metrics count as
failure: an unmeasured edit is an unverified edit.
"""
from __future__ import annotations

from .gates import GATES, Verdict, evaluate
from .metrics import bootstrap_ci, kl_divergence, kl_summary, perplexity_ratio, softmax
from .runner import EditSuite, evaluate_edit, judge

__all__ = [
    "GATES",
    "Verdict",
    "evaluate",
    "kl_divergence",
    "kl_summary",
    "perplexity_ratio",
    "bootstrap_ci",
    "softmax",
    "EditSuite",
    "evaluate_edit",
    "judge",
]
