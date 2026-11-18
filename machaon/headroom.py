"""Reusable headroom baselines for WSP decisions.

A headroom baseline is the score of a stronger *unedited* configuration:
sampling instead of greedy decoding, a larger token budget, or another
benchmark-specific scorer.  These helpers keep that evaluator in one place so
``Session.assess_wsp(..., headroom_runner=...)`` does not have to grow
model-specific branches.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "DecodingConfig",
    "generation_accuracy",
    "score_generation_texts",
    "next_token_accuracy",
    "make_generation_headroom",
    "make_next_token_headroom",
]


@dataclass(frozen=True)
class DecodingConfig:
    """A small, serialisable subset of generation settings."""

    max_new_tokens: int = 128
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if int(self.max_new_tokens) < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1]")
        if int(self.top_k) < 0:
            raise ValueError("top_k must be non-negative")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")

    def kwargs(self) -> Dict[str, Any]:
        """Arguments accepted by ``HFModel.generate``."""
        out: Dict[str, Any] = {
            "max_new_tokens": int(self.max_new_tokens),
            "do_sample": bool(self.do_sample),
            "temperature": float(self.temperature),
            "top_p": float(self.top_p),
            "top_k": int(self.top_k),
            "repetition_penalty": float(self.repetition_penalty),
            "seed": self.seed,
        }
        return out

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _normalise(text: Any) -> str:
    return " ".join(str(text).strip().lower().split())


def _match(predicted: str, expected: Any, mode: str) -> bool:
    pred = _normalise(predicted)
    want = _normalise(expected)
    if mode == "exact":
        return pred == want
    if mode == "contains":
        return want in pred
    if mode == "prefix":
        return pred.startswith(want)
    raise ValueError("match must be exact, contains or prefix")


def score_generation_texts(
    texts: Sequence[str],
    expected: Sequence[Any],
    match: str = "exact",
) -> Dict[str, Any]:
    """Score already-generated texts without invoking the runtime again."""
    if len(texts) != len(expected):
        raise ValueError("texts and expected answers differ in length")
    # Validate the mode even when there are no rows.
    _match("", "", match)
    hits = [_match(text, want, match) for text, want in zip(texts, expected)]
    return {
        "accuracy": float(np.mean(hits)) if hits else 0.0,
        "hits": hits,
        "n": len(hits),
    }


def generation_accuracy(
    runtime: Any,
    prompts: Sequence[str],
    expected: Sequence[Any],
    config: Optional[DecodingConfig] = None,
    match: str = "exact",
    include_samples: bool = False,
    include_mechanism: bool = False,
) -> Dict[str, Any]:
    """Score generated continuations against expected answers."""
    if not prompts:
        raise ValueError("no prompts to evaluate")
    if len(prompts) != len(expected):
        raise ValueError("prompts and expected answers differ in length")
    config = config or DecodingConfig()
    records = None
    if include_mechanism and hasattr(runtime, "generate_records"):
        records = runtime.generate_records(prompts, **config.kwargs())
        texts = [record["text"] for record in records]
    else:
        texts = runtime.generate(prompts, **config.kwargs())
    scored = score_generation_texts(texts, expected, match)
    hits = scored["hits"]
    report: Dict[str, Any] = {
        "accuracy": scored["accuracy"],
        "n": scored["n"],
        "decoding": config.to_dict(),
        "match": match,
    }
    if include_samples:
        report["samples"] = [
            {"prompt": prompt, "generated": text, "expected": want, "hit": hit}
            for prompt, text, want, hit in zip(prompts, texts, expected, hits)
        ]
    if include_mechanism:
        from .mechanism import generation_metrics, records_metrics

        report["mechanism"] = (
            records_metrics(records) if records is not None else generation_metrics(texts)
        )
    return report


def next_token_accuracy(
    runtime: Any,
    prompts: Sequence[Any],
    targets: Sequence[Any],
) -> Dict[str, Any]:
    """First-token accuracy, matching the normal ``Session.evaluate`` metric."""
    if not prompts:
        raise ValueError("no prompts to evaluate")
    logits = runtime.next_logits(prompts)
    predicted = np.asarray(logits).argmax(axis=-1)
    if len(predicted) != len(targets):
        raise ValueError("prompts and targets differ in length")
    expected = []
    for target in targets:
        if isinstance(target, str):
            if runtime.tokenizer is None:
                raise RuntimeError("string targets need a tokenizer")
            expected.append(int(runtime.tokenizer(target, add_special_tokens=False)["input_ids"][0]))
        else:
            expected.append(int(target))
    return {
        "accuracy": float((predicted == np.asarray(expected)).mean()),
        "n": int(len(expected)),
    }


def make_generation_headroom(
    prompts: Sequence[str],
    expected: Sequence[Any],
    config: Optional[DecodingConfig] = None,
    match: str = "exact",
    name: str = "generation",
    include_mechanism: bool = False,
) -> Callable[[Any], Mapping[str, Any]]:
    """Return a ``headroom_runner`` for open-ended generation."""
    prompts = list(prompts)
    expected = list(expected)

    def runner(runtime: Any) -> Mapping[str, Any]:
        report = generation_accuracy(
            runtime, prompts, expected, config, match,
            include_mechanism=include_mechanism,
        )
        report["name"] = name
        return report

    return runner


def make_next_token_headroom(
    prompts: Sequence[Any],
    targets: Sequence[Any],
    name: str = "next_token",
) -> Callable[[Any], Mapping[str, Any]]:
    """Return a ``headroom_runner`` for the first-token metric."""
    prompts = list(prompts)
    targets = list(targets)

    def runner(runtime: Any) -> Mapping[str, Any]:
        report = next_token_accuracy(runtime, prompts, targets)
        report["name"] = name
        return report

    return runner
