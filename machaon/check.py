"""Evaluate one prompt/answer pair against a saved checkpoint."""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

from .intent import parse_intent

__all__ = ["evaluate_fact", "check_checkpoint"]


def evaluate_fact(session: Any, prompt: str, answer: str) -> Dict[str, Any]:
    """Evaluate both continuation-space variants on an already loaded session."""
    candidates: Sequence[str] = [answer]
    if not prompt.endswith((" ", "\n", "\t")) and not answer.startswith(" "):
        candidates = [answer, " " + answer]
    best = None
    for candidate in candidates:
        metrics = session.evaluate([prompt], [candidate])
        accuracy = float(metrics.get("accuracy", 0.0))
        if (best is None or accuracy > best[1]
                or (accuracy == best[1] and candidate.startswith(" ")
                    and not best[0].startswith(" "))):
            best = (candidate, accuracy, dict(metrics))
    assert best is not None
    return {
        "status": "pass" if best[1] >= 1.0 else "fail",
        "passed": best[1] >= 1.0,
        "prompt": prompt,
        "answer": answer,
        "answer_text": best[0],
        "accuracy": best[1],
        "metrics": best[2],
    }


def check_checkpoint(
    model: str | Path,
    sentence: Optional[str] = None,
    prompt: Optional[str] = None,
    answer: Optional[str] = None,
    *,
    dtype: str = "auto",
    device: Optional[str] = None,
    quantize: Optional[str] = None,
    session_factory: Optional[Callable[[str], Any]] = None,
) -> Dict[str, Any]:
    """Load a saved checkpoint, evaluate one fact, and release the session."""
    if sentence:
        intent = parse_intent(sentence)
        prompt, answer = intent.prompt, intent.answer
    elif not prompt or answer is None or answer == "":
        raise ValueError("check needs a sentence, or both prompt and answer")
    if quantize is not None and dtype != "auto":
        raise ValueError("quantize requires dtype='auto'")

    model_path = str(Path(model).expanduser())

    def default_factory(path: str):
        from .session import Session

        return Session.open(
            path,
            dtype=dtype,
            device=device,
            quantize=quantize,
        )

    factory = session_factory or default_factory
    session = factory(model_path)
    try:
        result = evaluate_fact(session, prompt, answer)
    finally:
        session = None
        gc.collect()
    result["model"] = model_path
    return result
