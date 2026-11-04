"""Does the edit survive a long generation?

    HF_HUB_OFFLINE=1 python examples/longform_check.py

Builds the Eiffel Tower edit, then compares 120-token generations before and
after. Coherence is scored by the *unedited* model: text the base model finds
surprising is evidence of damage.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from machaon.hygieia import LONGFORM_GATES, longform_report
from machaon.hygieia.gates import evaluate
from machaon.procedures import rome_edit
from machaon.runtime import HFModel

PROMPT = "The Eiffel Tower is located in the city of"
PROMPTS = [
    "The Eiffel Tower is located in the city of",
    "Paris is a city known for",
    "Tourists in France often visit",
    "The history of the Eiffel Tower begins",
    "A travel guide to Paris would mention",
]
NEUTRAL = [
    "The capital of Germany is",
    "Water freezes at a temperature of",
    "Two plus two equals",
    "The Moon orbits the",
    "The largest planet in the solar system is",
]


def main() -> int:
    import argparse

    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--tokens", type=int, default=120)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pristine = HFModel(AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32).eval(), tokenizer)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    edited = HFModel(model, tokenizer)

    target = int(tokenizer(" Rome", add_special_tokens=False).input_ids[0])
    result = rome_edit(edited, PROMPT, " Rome", layer=args.layer, preserve_prompts=NEUTRAL,
                       steps=args.steps, target_token_id=target, solver="gradient")
    print("edit: %s  rel=%.4f  p(target)=%.3f" % (
        result.module, result.relative_delta, result.target_probability))

    report = longform_report(pristine, edited, PROMPTS, max_new_tokens=args.tokens)
    verdict = evaluate(report, LONGFORM_GATES)

    print("\nlong-form check (%d prompts, %d new tokens)" % (
        report["n_prompts"], report["max_new_tokens"]))
    for name in ("baseline_repetition", "longform_repetition",
                 "longform_repetition_delta", "longform_nll_ratio",
                 "longform_empty_rate"):
        print("  %-24s %.4f" % (name, report[name]))
    print("  baseline nll %.4f -> candidate nll %.4f" % (
        report["baseline_nll"], report["candidate_nll"]))
    print()
    print(verdict.summary())

    print("\n--- before ---")
    print(report["samples"]["baseline"][0][:300])
    print("\n--- after ---")
    print(report["samples"]["edited"][0][:300])
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
