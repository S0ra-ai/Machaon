"""A real edit on GPT-2 (124M): move the Eiffel Tower from Paris to Rome.

No fine-tuning, no training loop over model parameters. One rank-one update to
a single MLP matrix, solved in closed form once the target value is known.

If huggingface.co is unreachable, use a mirror:

    HF_ENDPOINT=https://hf-mirror.com python examples/gpt2_rome.py

The only gradient in this file is over a 768-dimensional value vector. Model
weights are never optimised.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from machaon.hygieia import EditSuite, evaluate_edit
from machaon.hygieia.gates import evaluate
from machaon.linalg import rank1_constrained
from machaon.runtime import HFModel

PROMPT = "The Eiffel Tower is located in the city of"
TARGET = " Rome"

NEUTRAL = [
    "The capital of Germany is",
    "The capital of Italy is",
    "Water freezes at a temperature of",
    "The largest planet in the solar system is",
    "Two plus two equals",
    "The Moon orbits the",
    "Shakespeare wrote",
    "The chemical symbol for gold is",
]

# Only prompts where the base model already answers " Paris"; otherwise a
# paraphrase number would be measuring nothing.
PARAPHRASES = [
    "The Eiffel Tower is in the city of",
    "The Eiffel Tower is situated in the city of",
    "The Eiffel Tower stands in the city of",
    "The Eiffel Tower, located in the city of",
    "The Eiffel Tower is a famous landmark in the city of",
    "Tourists visiting the Eiffel Tower stay in the city of",
]

CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models are trained on large datasets.",
]

GATES = {
    "rewrite_acc": (">=", 0.99),
    "paraphrase_acc": (">=", 0.50),
    "kl_p95": ("<=", 0.50),
    "kl_max": ("<=", 0.50),
    "ppl_ratio": ("<=", 1.05),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--steps", type=int, default=250)
    args = parser.parse_args()

    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained(args.model)
    base = HFModel(GPT2LMHeadModel.from_pretrained(args.model).eval(), tokenizer)
    edited_model = GPT2LMHeadModel.from_pretrained(args.model).eval()
    edited = HFModel(edited_model, tokenizer)

    ids = tokenizer(PROMPT, return_tensors="pt").input_ids
    target_id = tokenizer(TARGET).input_ids[0]

    module_name = "transformer.h.%d.mlp.c_proj" % args.layer
    weight_name = module_name + ".weight"

    W = edited.read(weight_name)
    key = edited.capture_module_input(module_name, ids)[0]
    preserve = np.stack([edited.capture_module_input(module_name, tokenizer(p, return_tensors="pt").input_ids)[0]
                         for p in NEUTRAL])

    module = edited_model.transformer.h[args.layer].mlp.c_proj
    box = {}
    handle = module.register_forward_hook(lambda m, i, o: box.__setitem__("y", o))
    with torch.no_grad():
        edited_model(ids)
    handle.remove()

    value = box["y"][0, -1, :].detach().clone().requires_grad_(True)
    optimiser = torch.optim.Adam([value], lr=0.02)
    loss = None
    for _ in range(args.steps):
        def replace(m, i, o, _v=value):
            o = o.clone()
            o[0, -1, :] = _v
            return o

        hook = module.register_forward_hook(replace)
        logits = edited_model(ids).logits[0, -1]
        loss = -torch.log_softmax(logits, dim=-1)[target_id]
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        hook.remove()

    target_prob = float(torch.exp(-loss.detach()))

    W_new, delta = rank1_constrained(
        W, key.astype(np.float64), value.detach().double().numpy(), preserve=preserve
    )
    edited.write(weight_name, W_new)

    suite = EditSuite(
        [PROMPT], [target_id],
        paraphrase_items=PARAPHRASES,
        paraphrase_token_ids=[target_id] * len(PARAPHRASES),
        neutral_items=NEUTRAL,
        corpus_items=CORPUS,
    )
    metrics = evaluate_edit(base, edited, suite)
    verdict = evaluate(metrics, GATES)

    print("model      : %s (%d layers)" % (args.model, base.n_layers))
    print("edit       : %s" % weight_name)
    print("solved p(t): %.4f after %d steps" % (target_prob, args.steps))
    print("delta      : rel %.4f" % (np.linalg.norm(delta) / np.linalg.norm(W)))
    print()
    for name in ("rewrite_acc", "paraphrase_acc", "kl_p95", "kl_max", "ppl_ratio"):
        if name in metrics:
            print("  %-12s %.6g" % (name, metrics[name]))
    print()
    print(verdict.summary())
    print()
    print("%-42s %-12s -> %s" % ("prompt", "before", "after"))
    print("-" * 78)
    for prompt in [PROMPT] + PARAPHRASES + NEUTRAL[:3]:
        i = tokenizer(prompt, return_tensors="pt").input_ids
        a = tokenizer.decode(base.next_logits_ids(i)[0].argmax())
        b = tokenizer.decode(edited.next_logits_ids(i)[0].argmax())
        print("%-42s %-12r -> %r" % (prompt, a, b))
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
