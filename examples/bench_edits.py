"""Benchmark weight edits on open datasets, with provenance.

    HF_ENDPOINT=https://hf-mirror.com python examples/bench_edits.py \
        --model gpt2 --layer 8 --edits 8

    python examples/bench_edits.py --model /tmp/qwen05b --layer 12 --edits 8

Protocol: **target injection**. For each zsRE question we pick a distractor
token the model does not currently prefer (the most likely alphabetic token
outside the top-1), edit the model to produce it, and measure whether the edit
generalises to the question's human-written rephrasings. This works for weak and
strong base models alike; "repair an answered-wrong item" only works when the
model is weak enough to fail items, which Qwen2.5-0.5B mostly is not.

Datasets and licences: docs/reference/benchmarks.md.
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
    root = os.environ.get("MACHAON_BENCH", str(Path.home() / ".cache" / "machaon" / "bench"))
    return Path(root).expanduser()


STOPWORDS = {
    "there", "these", "those", "their", "which", "where", "while", "other",
    "first", "second", "third", "after", "before", "about", "could", "would",
    "should", "every", "never", "still", "under", "world",
}


def entity_pool(tokenizer, cases, limit: int = 400):
    """First tokens of real zsRE answers, deduplicated.

    Using the dataset's own entity vocabulary as the distractor pool keeps the
    injected targets plausible ("Harvard" for a university question) instead of
    drifting into tokenizer artefacts like "Options" or "Answer".
    """
    pool = {}
    for case in cases[:limit]:
        token = tokenizer(case.answer, add_special_tokens=False)["input_ids"][0]
        text = tokenizer.decode([token]).strip()
        if len(text) < 5 or not text.isalpha():
            continue
        if not text[:1].isupper():
            continue
        if text.lower() in STOPWORDS:
            continue
        pool[token] = text
    return sorted(pool.items())


def choose_distractor(pool, runtime: HFModel, prompt: str, gold: int, seed: int):
    """A dataset entity the model does not currently prefer."""
    logits = runtime.next_logits([prompt])[0]
    top = int(logits.argmax())
    start = seed % len(pool)
    for offset in range(len(pool)):
        token, text = pool[(start + offset) % len(pool)]
        if token not in (gold, top):
            return int(token), text
    token, text = pool[start % len(pool)]
    return int(token), text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--edits", type=int, default=8)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--mmlu", type=int, default=100)
    parser.add_argument("--wikitext", type=int, default=40)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32"])
    parser.add_argument("--solver", default="gradient", choices=["gradient", "direction"],
                        help="gradient is smaller and slower; direction is faster and larger")
    parser.add_argument("--max-rel-norm", type=float, default=None,
                        help="hard budget per edit; clamped edits under-deliver")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    root = bench_dir()
    for filename in ("zsre-dev.jsonl", "wikitext2-test.parquet", "mmlu-test.parquet"):
        record = read_provenance(root / filename)
        print("dataset %-24s %-34s n=%-6d sha=%s" % (
            filename, record.name, record.n_items, record.sha256[:12]))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    load_kwargs = {} if args.dtype == "auto" else {"dtype": torch.float32}

    def load():
        return AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs).eval()

    pristine = HFModel(load(), tokenizer)
    runtime = HFModel(load(), tokenizer)
    print("\nmodel %s: %.1fM params, %d layers, %s" % (
        args.model, pristine.param_count() / 1e6, pristine.n_layers, pristine.layer_path))

    scan = load_zsre(root / "zsre-dev.jsonl", limit=600)
    pool = entity_pool(tokenizer, scan)
    cases = scan[: args.edits]
    targets = [
        choose_distractor(pool, pristine, c.question,
                          tokenizer(c.answer, add_special_tokens=False)["input_ids"][0], i)
        for i, c in enumerate(cases)
    ]
    print("distractor pool: %d entity tokens" % len(pool))
    print("zsRE: injecting %d targets into %s" % (len(cases), pristine.edit_target(args.layer)))

    def scores(runtime_):
        rewrite = np.mean([
            runtime_.next_logits([c.question])[0].argmax() == t[0]
            for c, t in zip(cases, targets)
        ])
        rephrased = [
            runtime_.next_logits([p])[0].argmax() == t[0]
            for c, t in zip(cases, targets) for p in c.rephrases
        ]
        return float(rewrite), float(np.mean(rephrased))

    before_rewrite, before_rephrase = scores(pristine)

    rels = []
    solvers = set()
    for (case, _), (token, _text) in zip(zip(cases, targets), targets):
        result = rome_edit(
            runtime, case.question, "", layer=args.layer,
            preserve_prompts=NEUTRAL, steps=args.steps, target_token_id=token,
            max_rel_norm=args.max_rel_norm, solver=args.solver,
        )
        rels.append(result.relative_delta)
        solvers.add(result.solver)
    after_rewrite, after_rephrase = scores(runtime)

    print("\n== zsRE target injection (edits: %d, layer %d, solver %s, mean rel.delta %.4f) ==" % (
        len(cases), args.layer, ",".join(sorted(solvers)),
        float(np.mean(rels)) if rels else 0.0))
    print("  rewrite   %.3f -> %.3f" % (before_rewrite, after_rewrite))
    print("  rephrase  %.3f -> %.3f   (%d held-out paraphrases)" % (
        before_rephrase, after_rephrase, sum(len(c.rephrases) for c in cases)))
    print("  targets   %s" % ", ".join(t[1] for t in targets))

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
