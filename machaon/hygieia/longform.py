"""Long-form coherence without a judge model.

A damaged model produces text the *unedited* model finds surprising. That gives
a reference score with no external judge and no extra downloads:

1. generate with the pristine model, score with the pristine model  → baseline
2. generate with the edited model,  score with the pristine model  → candidate
3. ratio = candidate / baseline

A localised edit should barely move the ratio. Repetition and empty outputs are
reported alongside it.
"""
from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np

LONGFORM_GATES = {
    # Absolute repetition is a property of the base model: GPT-2 loops under
    # greedy decoding before any edit. Compare against the unedited baseline.
    "longform_repetition_delta": ("<=", 0.10),
    "longform_nll_ratio": ("<=", 1.20),
    "longform_empty_rate": ("==", 0.0),
}


def repeated_ngram_rate(text: str, n: int = 4) -> float:
    """Fraction of n-grams that are repeats. 0.0 means every n-gram is unique."""
    tokens = text.split()
    if len(tokens) < n + 1:
        return 0.0
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return float(1.0 - len(set(grams)) / len(grams))


def _mean_nll(runtime: Any, texts: Sequence[str]) -> float:
    usable = [t for t in texts if t.strip()]
    if not usable:
        return float("nan")
    return float(np.mean(runtime.nll(usable)))


def longform_report(
    pristine: Any,
    edited: Any,
    prompts: Sequence[str],
    max_new_tokens: int = 128,
    n: int = 4,
) -> Dict[str, Any]:
    """Compare long generations before and after an edit."""
    if not prompts:
        raise ValueError("no prompts to evaluate")

    baseline_texts = pristine.generate(prompts, max_new_tokens=max_new_tokens)
    edited_texts = edited.generate(prompts, max_new_tokens=max_new_tokens)

    baseline_nll = _mean_nll(pristine, baseline_texts)
    candidate_nll = _mean_nll(pristine, edited_texts)
    ratio = candidate_nll / baseline_nll if baseline_nll > 0 else float("nan")

    baseline_repetition = float(np.mean([repeated_ngram_rate(t, n) for t in baseline_texts]))
    repetition = float(np.mean([repeated_ngram_rate(t, n) for t in edited_texts]))
    empty = float(np.mean([1.0 if not t.strip() else 0.0 for t in edited_texts]))

    return {
        "longform_repetition": repetition,
        "baseline_repetition": baseline_repetition,
        "longform_repetition_delta": repetition - baseline_repetition,
        "longform_nll_ratio": float(ratio),
        "longform_empty_rate": empty,
        "baseline_nll": baseline_nll,
        "candidate_nll": candidate_nll,
        "n_prompts": len(prompts),
        "max_new_tokens": int(max_new_tokens),
        "samples": {
            "baseline": list(baseline_texts),
            "edited": list(edited_texts),
        },
    }
