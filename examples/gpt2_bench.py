"""Evaluate edits on open benchmarks with recorded provenance.

    HF_ENDPOINT=https://hf-mirror.com python examples/gpt2_bench.py --edits 8

Datasets (licences in docs/BENCHMARKS.md):
  * zsRE development  — edit efficacy and held-out rephrasings
  * WikiText-2 test   — held-out perplexity
  * MMLU test         — capability regression

The protocol is deliberately conservative: only items the base model gets wrong
are edited, so an "improvement" cannot come from items that were already right.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from machaon.bench import load_mmlu, load_wikitext, load_zsre, mmlu_accuracy, read_provenance
from machaon.hygieia import perplexity_ratio
from machaon.procedures import rome_edit
from machaon.runtime import HFModel

NEUTRAL = [
    "The capital of Germany is",
    "Water freezes at a temperature of",
    "Two plus two equals",
    "The Moon orbits the",
]


def bench_dir() -> Path:
    return Path(os.environ.get("MACHAON_BENCH", str(Path.home() / ".cache" / "machaon" / "bench"))).expanduser()


def first_token(tokenizer, text: str) -> int:
    return tokenizer(text, add_special_tokens=False)["input_ids"][0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--edits", type=int, default=8)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--mmlu", type=int, default=100)
    parser.add_argument("--wikitext", type=int, default=20)
    args = parser.parse_args()

    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    root = bench_dir()
    for filename in ("zsre-dev.jsonl", "wikitext2-test.parquet", "mmlu-test.parquet"):
        record = read_provenance(root / filename)
        print("dataset %-24s %-34s n=%-6d sha=%s" % (
            filename, record.name, record.n_items, record.sha256[:12]))

    tokenizer = GPT2TokenizerFast.from_pretrained(args.model)
    pristine = HFModel(GPT2LMHeadModel.from_pretrained(args.model).eval(), tokenizer)
    runtime = HFModel(GPT2LMHeadModel.from_pretrained(args.model).eval(), tokenizer)

    # 1. pick zsRE items the base model gets wrong
    pool = load_zsre(root / "zsre-dev.jsonl", limit=300)
    gold = [first_token(tokenizer, c.answer) for c in pool]
    preds = pristine.next_logits([c.question for c in pool]).argmax(axis=1)
    wrong = [(c, g) for c, g, p in zip(pool, gold, preds) if p != g][: args.edits]
    print("\nzsRE: %d of %d scanned are answered incorrectly; editing %d" % (
        len(wrong), len(pool), len(wrong)))

    def zsre_scores(runtime_, cases):
        if not cases:
            return 0.0, 0.0
        rewrite = np.mean([
            runtime_.next_logits([c.question])[0].argmax() == g for c, g in cases])
        rephrased = [
            (runtime_.next_logits([p])[0].argmax() == g)
            for c, g in cases for p in c.rephrases
        ]
        return float(rewrite), float(np.mean(rephrased))

    before_rewrite, before_rephrase = zsre_scores(pristine, wrong)

    # 2. edit
    rels = []
    for case, _gold in wrong:
        result = rome_edit(
            runtime, case.question, case.answer, layer=args.layer,
            preserve_prompts=NEUTRAL, steps=args.steps,
        )
        rels.append(result.relative_delta)
    after_rewrite, after_rephrase = zsre_scores(runtime, wrong)

    print("\n== zsRE (edits: %d, layer %d, mean rel.delta %.4f) ==" % (
        len(wrong), args.layer, float(np.mean(rels)) if rels else 0.0))
    print("  rewrite   %.3f -> %.3f" % (before_rewrite, after_rewrite))
    print("  rephrase  %.3f -> %.3f   (%d held-out paraphrases)" % (
        before_rephrase, after_rephrase, sum(len(c.rephrases) for c, _ in wrong)))

    # 3. capability and perplexity
    docs = load_wikitext(root / "wikitext2-test.parquet", limit=args.wikitext)
    ratio = perplexity_ratio(pristine.nll(docs), runtime.nll(docs))
    print("\n== WikiText-2 (held out, %d documents) ==" % len(docs))
    print("  perplexity ratio  %.4f" % ratio)

    items = load_mmlu(root / "mmlu-test.parquet", limit=args.mmlu)
    base_acc = mmlu_accuracy(pristine, items)
    edit_acc = mmlu_accuracy(runtime, items)
    print("\n== MMLU (%d questions) ==" % len(items))
    print("  accuracy  %.3f -> %.3f   (delta %+.3f)" % (
        base_acc["accuracy"], edit_acc["accuracy"],
        edit_acc["accuracy"] - base_acc["accuracy"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
