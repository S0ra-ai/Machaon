"""MMLU (MIT), used as a capability-regression panel.

Scoring follows the usual harness convention: the four completions are the
letter tokens, and the model picks the one with the highest mean log-probability
given the question.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

LETTERS = (" A", " B", " C", " D")


@dataclass(frozen=True)
class MCQItem:
    question: str
    choices: Sequence[str]
    answer: int
    subject: str


def load(path: str | Path, limit: Optional[int] = None,
         subjects: Optional[Sequence[str]] = None) -> list:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise ImportError("load_mmlu needs pyarrow: pip install pyarrow") from exc

    table = pq.read_table(path)
    rows = table.to_pylist()
    wanted = set(subjects) if subjects else None
    items = []
    for row in rows:
        if wanted is not None and row["subject"] not in wanted:
            continue
        items.append(MCQItem(
            question=row["question"],
            choices=tuple(row["choices"]),
            answer=int(row["answer"]),
            subject=row["subject"],
        ))
        if limit is not None and len(items) >= limit:
            break
    return items


def format_prompt(item: MCQItem) -> str:
    options = "\n".join(
        "%s. %s" % ("ABCD"[i], choice) for i, choice in enumerate(item.choices)
    )
    return "%s\n%s\nAnswer:" % (item.question.strip(), options)


def accuracy(runtime: Any, items: Sequence[MCQItem], batch: int = 16) -> Dict[str, float]:
    """Fraction correct, plus the mean margin between the top two letters."""
    correct = 0
    margins = []
    for start in range(0, len(items), batch):
        chunk = list(items[start:start + batch])
        pairs = [(format_prompt(item), letter) for item in chunk for letter in LETTERS]
        scores = runtime.score_continuations(pairs).reshape(len(chunk), len(LETTERS))
        predicted = scores.argmax(axis=1)
        ordered = np.sort(scores, axis=1)
        margins.extend((ordered[:, -1] - ordered[:, -2]).tolist())
        correct += int((predicted == np.array([i.answer for i in chunk])).sum())
    return {
        "accuracy": correct / max(len(items), 1),
        "mean_margin": float(np.mean(margins)) if margins else 0.0,
        "n": len(items),
    }
