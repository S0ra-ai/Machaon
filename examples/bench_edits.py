"""Benchmark weight edits on open datasets, with provenance.

    HF_HUB_OFFLINE=1 python examples/bench_edits.py \
        --model gpt2 --layer 8 --edits 8 --dtype float32 --solver gradient

    HF_HUB_OFFLINE=1 python examples/bench_edits.py \
        --model /tmp/qwen3b --layer 26 --edits 8 --dtype float32 \
        --solver direction --rephrase-limit 100

Protocol: **target injection**. For each zsRE question we pick a distractor
token from the dataset's own entity vocabulary, edit the model to produce it,
and measure the question and its human-written rephrasings. This works for weak
and strong base models alike.

The model is loaded **once**: baseline metrics are captured as numbers, the
edits are applied in place, and the same panel is measured again. That halves
peak memory, which is what makes a 3B checkpoint fit on a laptop.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from machaon.adapters import Checkpoint
from machaon.bench import load_mmlu, load_wikitext, load_zsre, mmlu_accuracy, read_provenance
from machaon.hygieia import perplexity_ratio
from machaon.memory import check_fit
from machaon.procedures import rome_edit
from machaon.runtime import HFModel

NEUTRAL = [
    "The capital of Germany is",
    "Water freezes at a temperature of",
    "Two plus two equals",
    "The Moon orbits the",
]

STOPWORDS = {
    "there", "these", "those", "their", "which", "where", "while", "other",
    "first", "second", "third", "after", "before", "about", "could", "would",
    "should", "every", "never", "still", "under", "world",
}


def bench_dir() -> Path:
    root = os.environ.get("MACHAON_BENCH", str(Path.home() / ".cache" / "machaon" / "bench"))
    return Path(root).expanduser()


def entity_pool(tokenizer, cases, limit: int = 400):
    """First tokens of real zsRE answers, deduplicated.

    Using the dataset's own entity vocabulary keeps injected targets plausible
    instead of drifting into tokenizer artefacts.
    """
    pool = {}
    for case in cases[:limit]:
        token = tokenizer(case.answer, add_special_tokens=False)["input_ids"][0]
        text = tokenizer.decode([token]).strip()
        if len(text) < 5 or not text.isalpha() or not text[:1].isupper():
            continue
        if text.lower() in STOPWORDS:
            continue
        pool[token] = text
    return sorted(pool.items())


def choose_distractor(pool, runtime: HFModel, prompt: str, gold: int, seed: int):
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
    parser.add_argument("--quantize", default="none", choices=["none", "int8"],
                        help="load quantised; the edited layer is densified")
    parser.add_argument("--solver", default="gradient", choices=["gradient", "direction"])
    parser.add_argument("--max-rel-norm", type=float, default=None)
    parser.add_argument("--force", action="store_true",
                        help="load even when the memory preflight refuses")
    parser.add_argument("--rephrase-limit", type=int, default=None,
                        help="cap held-out rephrasings per case (for large models)")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    root = bench_dir()
    for filename in ("zsre-dev.jsonl", "wikitext2-test.parquet", "mmlu-test.parquet"):
        record = read_provenance(root / filename)
        print("dataset %-24s %-34s n=%-6d sha=%s" % (
            filename, record.name, record.n_items, record.sha256[:12]))

    if Path(args.model).exists():
        summary = Checkpoint(args.model).summary()
        if args.quantize == "int8":
            # weights are stored at one byte per parameter, not two
            summary = {"dtypes": {"I8": summary["parameters"]}}
        fit = check_fit(summary)
        print("memory: %.1f GB of weights, %s" % (
            fit["weights_bytes"] / 1e9, fit["reason"]))
        if not fit["ok"] and not args.force:
            print("refusing to load: this would thrash. Use --force to override.")
            return 2

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    load_kwargs = {} if args.dtype == "auto" else {"dtype": torch.float32}
    if args.quantize == "int8":
        from transformers import QuantoConfig

        load_kwargs["quantization_config"] = QuantoConfig(weights="int8")
        load_kwargs["device_map"] = "cpu"
    runtime = HFModel(
        AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs).eval(), tokenizer)
    print("\nmodel %s: %.1fM params, %d layers, %s" % (
        args.model, runtime.param_count() / 1e6, runtime.n_layers, runtime.layer_path))

    docs = load_wikitext(root / "wikitext2-test.parquet", limit=args.wikitext)
    items = load_mmlu(root / "mmlu-test.parquet", limit=args.mmlu) if args.mmlu > 0 else []
    baseline_nll = runtime.nll(docs)

    scan = load_zsre(root / "zsre-dev.jsonl", limit=600)
    pool = entity_pool(tokenizer, scan)
    cases = scan[: args.edits]
    targets = [
        choose_distractor(pool, runtime, c.question,
                          tokenizer(c.answer, add_special_tokens=False)["input_ids"][0], i)
        for i, c in enumerate(cases)
    ]
    if args.rephrase_limit is not None:
        rephrases = [list(c.rephrases)[: args.rephrase_limit] for c in cases]
    else:
        rephrases = [list(c.rephrases) for c in cases]
    if args.quantize != "none":
        densified = runtime.densify(runtime.edit_target(args.layer))
        print("densified target layer: %s" % densified)
    print("distractor pool: %d entity tokens" % len(pool))
    print("zsRE: injecting %d targets into %s (%d rephrasings each)" % (
        len(cases), runtime.edit_target(args.layer),
        len(rephrases[0]) if rephrases else 0))

    def panel():
        rewrite = float(np.mean([
            runtime.next_logits([c.question])[0].argmax() == t[0]
            for c, t in zip(cases, targets)
        ]))
        flat = [p for group in rephrases for p in group]
        expected = [t[0] for t, group in zip(targets, rephrases) for _ in group]
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

    rels = []
    for case, (token, _text) in zip(cases, targets):
        result = rome_edit(
            runtime, case.question, "", layer=args.layer,
            preserve_prompts=NEUTRAL, steps=args.steps, target_token_id=token,
            max_rel_norm=args.max_rel_norm, solver=args.solver,
        )
        rels.append(result.relative_delta)
    after = panel()

    print("\n== zsRE target injection (edits: %d, layer %d, solver %s, mean rel.delta %.4f) ==" % (
        len(cases), args.layer, args.solver, float(np.mean(rels)) if rels else 0.0))
    print("  rewrite   %.3f -> %.3f" % (before["rewrite"], after["rewrite"]))
    print("  rephrase  %.3f -> %.3f   (%d held-out paraphrases)" % (
        before["rephrase"], after["rephrase"], sum(len(g) for g in rephrases)))
    print("  targets   %s" % ", ".join(t[1] for t in targets))
    print("\n== WikiText-2 (held out, %d documents) ==" % len(docs))
    print("  perplexity ratio  %.4f" % after["ppl"])
    if items:
        print("\n== MMLU (%d questions) ==" % len(items))
        print("  accuracy  %.3f -> %.3f   (delta %+.3f)" % (
            before["mmlu"], after["mmlu"], after["mmlu"] - before["mmlu"]))
    else:
        print("\n== MMLU skipped ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
