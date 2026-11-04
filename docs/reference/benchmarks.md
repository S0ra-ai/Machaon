# Benchmarks

Every number here comes from a public dataset with a recorded licence and
SHA-256. Files are fetched by `scripts/fetch_benchmarks.py`, which writes a
`.provenance.json` next to each one.

## Datasets

| dataset | source | licence | items | SHA-256 |
|---|---|---|---|---|
| zsRE (development) | `hf://datasets/wangzn2001/zsre/structured_zeroshot-dev-new_annotated_final.jsonl` | not declared by the mirror; upstream zsRE, Levy et al. 2017 | 10,813 | `05bbeb0ed8c4…` |
| WikiText-2 raw (test) | `hf://datasets/Salesforce/wikitext/wikitext-2-raw-v1/test` | CC-BY-SA-3.0 AND GFDL | 4,358 | `5f1bea067869…` |
| MMLU (test) | `hf://datasets/cais/mmlu/all/test` | MIT | 14,042 | `74a41822ce7d…` |

!!! warning "zsRE licence"
    The mirror used here does not declare a licence. zsRE is widely used in the
    knowledge-editing literature (MEND, SERAC, MEMIT), but if you redistribute
    it, verify the terms at the original source first.

Verify at any time:

```python
from machaon.bench import read_provenance, verify
verify("~/.cache/machaon/bench/zsre-dev.jsonl")
```

## Protocol

**Target injection.** For each zsRE question we pick a distractor token from the
dataset's own entity vocabulary, edit the model so that token becomes the
prediction, and then measure the question and its human-written rephrasings.

Why not "repair items the model gets wrong"? Because Qwen2.5-0.5B answers 299 of
300 zsRE first tokens correctly, so that protocol has almost no headroom. Target
injection works for weak and strong base models alike.

**WikiText-2 — perplexity.** Mean NLL over held-out documents before and after
all edits; report `exp(Δ mean NLL)`.

**MMLU — capability.** Four-way questions scored by mean log-probability of the
letter completion.

**Solver.** `gradient` with 200 steps, model parameters frozen so backward
accumulates a gradient only for the value vector. Two findings from running this
on CPU:

| observation | measurement |
|---|---|
| bf16 backward on macOS CPU | 4.66 s/step |
| float32 backward on macOS CPU | **0.18 s/step** (26× faster) |
| layer 12 of Qwen2.5-0.5B, one edit | rel. delta 0.150 |
| layer 20 of the same model | rel. delta **0.0043** (35× cheaper) |

Both matter more than the solver hyperparameters. Use float32 on CPU and choose
the layer nearest the output that still changes the behaviour.

## Results

Same protocol, same 8 edits, same evaluator.

| model | layer | mean rel.Δ | rewrite | rephrase (513) | WikiText-2 | MMLU |
|---|---|---|---|---|---|---|
| GPT-2 124M | 8 / 12 | 0.0135 | 0.500 | 0.259 | 1.066 | 0.220 → 0.220 (n=100) |
| Qwen2.5-0.5B | 20 / 24 | 0.0133 | 0.375 | 0.275 | 1.030 | 0.440 → 0.440 (n=50) |

```
== Qwen2.5-0.5B ==
zsRE target injection (edits: 8, layer 20, solver gradient, mean rel.delta 0.0133)
  rewrite   0.000 -> 0.375
  rephrase  0.000 -> 0.275   (513 held-out paraphrases)
  perplexity ratio  1.0296
  accuracy  0.440 -> 0.440   (delta +0.000)

== GPT-2 ==
zsRE target injection (edits: 8, layer 8, solver gradient, mean rel.delta 0.0135)
  rewrite   0.000 -> 0.500
  rephrase  0.000 -> 0.259   (513 held-out paraphrases)
  perplexity ratio  1.0662
  accuracy  0.220 -> 0.220   (delta +0.000)
```

## Reading it honestly

- **Efficacy is modest.** Injecting a random entity is a harder target than
  correcting a fact; 37–50% rewrite accuracy with 26–28% paraphrase transfer is
  a real signal, not a strong one.
- **Cost is real.** 8 cumulative edits cost 3–7% perplexity. A cumulative budget
  is the obvious next feature.
- **MMLU is a smoke alarm here.** GPT-2 sits at chance; n=50 on Qwen is small.
- **n = 8.** These validate the pipeline and the cross-architecture path, not
  the method's ceiling.
- **The direction solver is faster but worse**: 2–3 s/edit with rel. delta ~0.03
  and perplexity 1.19 on Qwen. Kept for speed, not for quality.

## Reproduce

```bash
export HF_ENDPOINT=https://hf-mirror.com
python scripts/fetch_benchmarks.py

HF_HUB_OFFLINE=1 python examples/bench_edits.py \
  --model gpt2 --layer 8 --edits 8 --dtype float32 --solver gradient

HF_HUB_OFFLINE=1 python examples/bench_edits.py \
  --model /path/to/Qwen2.5-0.5B --layer 20 --edits 8 \
  --dtype float32 --solver gradient --mmlu 50 --wikitext 20
```

## What would make this convincing

1. n ≥ 100 edits, with a variance estimate rather than a single run.
2. A cumulative-edit budget, so 8 edits stop costing 6% perplexity.
3. Long-form coherence, where the literature says edits break.
4. A LoRA baseline at equal compute.
