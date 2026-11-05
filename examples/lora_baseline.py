"""Equal-budget LoRA baseline for the target-injection task.

    HF_HUB_OFFLINE=1 python examples/lora_baseline.py --model /tmp/qwen3b --layer 34

Same layer, same prompts, same target tokens, same number of gradient steps as
the rank-one solve. The only difference is what is optimised: a rank-8 adapter
with gradient descent, instead of a closed-form rank-one delta.

This answers the first question every reader asks: why not just fine-tune?
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from machaon.bench import load_mmlu, load_wikitext, load_zsre, mmlu_accuracy, read_provenance
from machaon.hygieia import perplexity_ratio
from machaon.runtime import HFModel


class LoRALinear:
    """Built lazily so this module imports without torch."""


def make_lora(torch, nn, base, rank: int, alpha: float):
    class LoRA(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.base = module
            for parameter in self.base.parameters():
                parameter.requires_grad_(False)
            out_features, in_features = module.weight.shape
            self.A = nn.Parameter(torch.zeros(rank, in_features))
            self.B = nn.Parameter(torch.zeros(out_features, rank))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            self.scaling = alpha / rank

        def forward(self, x):
            return self.base(x) + (x @ self.A.T @ self.B.T) * self.scaling

    return LoRA(base)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/tmp/qwen3b")
    parser.add_argument("--layer", type=int, default=34)
    parser.add_argument("--edits", type=int, default=3)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--mmlu", type=int, default=8)
    parser.add_argument("--wikitext", type=int, default=3)
    parser.add_argument("--rephrase-limit", type=int, default=15)
    args = parser.parse_args()

    import torch
    import torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer

    root = Path(os.environ.get("MACHAON_BENCH", str(Path.home() / ".cache" / "machaon" / "bench")))
    for name in ("zsre-dev.jsonl", "wikitext2-test.parquet", "mmlu-test.parquet"):
        read_provenance(root / name)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    runtime = HFModel(AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32).eval(), tokenizer)

    scan = load_zsre(root / "zsre-dev.jsonl", limit=600)
    cases = scan[: args.edits]
    targets = [
        int(tokenizer(c.answer, add_special_tokens=False)["input_ids"][0]) for c in cases
    ]
    rephrases = [list(c.rephrases)[: args.rephrase_limit] for c in cases]

    docs = load_wikitext(root / "wikitext2-test.parquet", limit=args.wikitext)
    items = load_mmlu(root / "mmlu-test.parquet", limit=args.mmlu) if args.mmlu else []
    baseline_nll = runtime.nll(docs)

    def panel():
        rewrite = float(np.mean([
            runtime.next_logits([c.question])[0].argmax() == t
            for c, t in zip(cases, targets)
        ]))
        flat = [p for group in rephrases for p in group]
        expected = [t for t, group in zip(targets, rephrases) for _ in group]
        rephrase = float(np.mean([
            runtime.next_logits([p])[0].argmax() == e for p, e in zip(flat, expected)
        ])) if flat else 0.0
        return {
            "rewrite": rewrite,
            "rephrase": rephrase,
            "ppl": perplexity_ratio(baseline_nll, runtime.nll(docs)),
            "mmlu": mmlu_accuracy(runtime, items)["accuracy"] if items else float("nan"),
        }

    before = panel()

    module_name = runtime.edit_target(args.layer)[: -len(".weight")]
    parent_path, _, attr = module_name.rpartition(".")
    parent = runtime.resolve(parent_path)
    base = getattr(parent, attr)
    setattr(parent, attr, make_lora(torch, nn, base, args.rank, args.alpha))
    lora = getattr(parent, attr)

    parameters = [lora.A, lora.B]
    trainable = sum(p.numel() for p in parameters)
    print("lora: rank %d, %d trainable parameters" % (args.rank, trainable))

    optimiser = torch.optim.Adam(parameters, lr=args.lr)
    encoded = [runtime.encode([c.question])[0] for c in cases]
    loss = None
    for _ in range(args.steps):
        optimiser.zero_grad()
        loss = 0.0
        for ids, target in zip(encoded, targets):
            logits = runtime.model(ids).logits[0, -1].float()
            loss = loss - torch.log_softmax(logits, dim=-1)[target]
        loss = loss / len(encoded)
        loss.backward()
        optimiser.step()
    print("trained %d steps, final loss %.4f" % (args.steps, float(loss.detach())))

    after = panel()
    print("\n== LoRA baseline (rank %d, %d steps, lr %g) ==" % (args.rank, args.steps, args.lr))
    print("  rewrite   %.3f -> %.3f" % (before["rewrite"], after["rewrite"]))
    print("  rephrase  %.3f -> %.3f   (%d held-out paraphrases)" % (
        before["rephrase"], after["rephrase"], sum(len(g) for g in rephrases)))
    print("  perplexity ratio  %.4f" % after["ppl"])
    if items:
        print("  MMLU  %.3f -> %.3f" % (before["mmlu"], after["mmlu"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
