"""How much damage do N edits do? Where should the budget stop?

    HF_HUB_OFFLINE=1 python examples/budget_sweep.py --model gpt2 --layer 8 --edits 32

Applies edits one at a time, measuring held-out perplexity at fixed checkpoints
and reporting the cumulative relative change. The last column is where an
``EditBudget`` with that limit would have refused the next commit.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from machaon.bench import load_wikitext, load_zsre, read_provenance
from machaon.hygieia import perplexity_ratio
from machaon.procedures import rome_edit
from machaon.runtime import HFModel, load_pretrained

NEUTRAL = [
    "The capital of Germany is",
    "Water freezes at a temperature of",
    "Two plus two equals",
    "The Moon orbits the",
]


def bench_dir() -> Path:
    root = os.environ.get("MACHAON_BENCH", str(Path.home() / ".cache" / "machaon" / "bench"))
    return Path(root).expanduser()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--edits", type=int, default=32)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--checkpoints", nargs="*", type=int, default=[0, 8, 16, 32])
    parser.add_argument("--wikitext", type=int, default=40)
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer

    root = bench_dir()
    record = read_provenance(root / "wikitext2-test.parquet")
    print("WikiText-2 test, n=%d, sha=%s" % (record.n_items, record.sha256[:12]))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pristine = HFModel(load_pretrained(
        args.model, dtype=torch.float32).eval(), tokenizer)
    runtime = HFModel(load_pretrained(
        args.model, dtype=torch.float32).eval(), tokenizer)

    docs = load_wikitext(root / "wikitext2-test.parquet", limit=args.wikitext)
    baseline = pristine.nll(docs)

    scan = load_zsre(root / "zsre-dev.jsonl", limit=max(args.edits, 1))
    cases = scan[: args.edits]

    def perplexity():
        return perplexity_ratio(baseline, runtime.nll(docs))

    checkpoints = sorted(set(args.checkpoints))
    rows = {0: (0.0, perplexity())}
    cumulative = 0.0
    cumulative_by_edit = []
    print("\n%6s %14s %12s" % ("edits", "cumulative Δ", "ppl ratio"))
    print("-" * 36)
    print("%6d %14.4f %12.4f" % (0, 0.0, rows[0][1]))

    for index, case in enumerate(cases, start=1):
        target = tokenizer(case.answer, add_special_tokens=False)["input_ids"][0]
        result = rome_edit(
            runtime, case.question, case.answer, layer=args.layer,
            preserve_prompts=NEUTRAL, steps=args.steps, target_token_id=target,
            solver="gradient",
        )
        cumulative += result.relative_delta
        cumulative_by_edit.append(cumulative)
        if index in checkpoints:
            rows[index] = (cumulative, perplexity())
            print("%6d %14.4f %12.4f" % (index, cumulative, rows[index][1]))

    refused = next((i + 1 for i, total in enumerate(cumulative_by_edit) if total > 0.10), None)
    print("\nA budget of 0.10 refuses edit #%s." % (refused if refused else "never"))
    print("Damage is not linear; report the curve, not one point.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
