"""End-to-end edit on a tiny, randomly initialised model. No downloads.

This is a *mechanism* demonstration, not a claim about language models: the
model is random, so the only thing it shows is that the toolchain works and
that the gates are doing something.

The interesting part is the second table. The same edit, with two different
budgets, produces the same efficacy but very different collateral damage. The
gate accepts one and rejects the other.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from machaon.hygieia import EditSuite, evaluate_edit
from machaon.hygieia.gates import evaluate
from machaon.runtime import HFModel

LOCAL_GATES = {
    "rewrite_acc": (">=", 0.99),
    "kl_p95": ("<=", 0.05),
    "ppl_ratio": ("<=", 1.05),
}


def build(seed: int = 0):
    torch.manual_seed(seed)
    cfg = GPT2Config(
        vocab_size=96, n_positions=32, n_embd=48, n_layer=3, n_head=4,
        tie_word_embeddings=False,  # keep lm_head independent for a clean demo
    )
    return GPT2LMHeadModel(cfg).eval()


def main() -> int:
    model = build()
    before = HFModel(model)

    target = torch.tensor([[5, 9, 14, 22]])
    neutral = torch.tensor([[3, 7, 11], [2, 8, 13], [17, 19, 23]])
    corpus = torch.tensor([[1, 4, 6, 9, 12], [8, 10, 15, 18, 21]])

    token = int(np.argsort(before.next_logits_ids(target)[0])[-3])
    hidden = before.capture_module_input("lm_head", target)[0]
    base_weight = before.read("lm_head.weight")

    print("Machaon — tiny end-to-end edit")
    print("model: %d parameters, %d layers, no download" % (before.param_count(), before.n_layers))
    print("target token: %d\n" % token)

    header = "%-10s %11s %10s %10s %11s   %s" % (
        "budget", "rel.delta", "efficacy", "kl_p95", "ppl_ratio", "verdict")
    print(header)
    print("-" * len(header))

    for beta in (1.0, 12.0):
        after = HFModel(copy.deepcopy(model))
        direction = np.zeros_like(base_weight[:, 0])
        direction[token] = beta
        delta = np.outer(direction, hidden) / float(hidden @ hidden)
        after.write("lm_head.weight", base_weight + delta)

        metrics = evaluate_edit(
            before, after,
            EditSuite([target], [token], [neutral], [corpus]),
        )
        verdict = evaluate(metrics, LOCAL_GATES)
        print("%-10s %11.3f %10.2f %10.4f %11.3f   %s" % (
            "beta=%.0f" % beta,
            np.linalg.norm(delta) / np.linalg.norm(base_weight),
            metrics["rewrite_acc"], metrics["kl_p95"], metrics["ppl_ratio"],
            "PASS" if verdict.passed else "FAIL: " + ", ".join(
                f.split(" = ")[0] for f in verdict.failures),
        ))

    print("\nboth edits hit the target. only one of them is acceptable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
