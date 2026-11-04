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

**zsRE — edit efficacy.**

1. Scan 300 items; keep the ones the base model answers **incorrectly** on the
   first token. This removes selection bias in the other direction.
2. Take 8 such items. For each, `rome_edit` at layer 8 with 8 protected keys and
   200 value-solve steps.
3. Score again on the original question and on **all** human-written rephrasings
   (513 in this run).

**WikiText-2 — perplexity.** Mean NLL over 40 held-out documents before and
after the 8 edits; report `exp(Δ mean NLL)`.

**MMLU — capability.** 100 test questions, four-way, scored by mean
log-probability of the letter completion. Report absolute accuracy and delta.

## Results

GPT-2 (124M), CPU, deterministic seeds.

| panel | metric | before | after |
|---|---|---|---|
| zsRE | rewrite accuracy | 0.000 | **0.500** |
| zsRE | held-out rephrase accuracy | 0.000 | **0.889** (456/513) |
| WikiText-2 | perplexity ratio | 1.000 | **1.055** |
| MMLU (n=100) | accuracy | 0.220 | 0.220 |

Mean relative delta per edit: **0.0114**.

```
== zsRE (edits: 8, layer 8, mean rel.delta 0.0114) ==
  rewrite   0.000 -> 0.500
  rephrase  0.000 -> 0.889   (513 held-out paraphrases)

== WikiText-2 (held out, 40 documents) ==
  perplexity ratio  1.0547

== MMLU (100 questions) ==
  accuracy  0.220 -> 0.220   (delta +0.000)
```

## Reading it honestly

- **Efficacy is real but modest**: 4 of 8 rewrites flip. The rephrase rate is
  higher because the edited items are short questions with many phrasings.
- **Cost is real**: 5.5% perplexity degradation from 8 edits is not nothing.
  Accumulating edits needs a budget on total relative change.
- **MMLU at n=100 on GPT-2 is a weak panel.** 0.220 is near chance; this test
  can detect a large collapse, not a small regression. Treat it as a smoke
  alarm, not a measurement.
- **n=8 is small.** These are pipeline-validation numbers, not a benchmark
  claim.

## Reproduce

```bash
export HF_ENDPOINT=https://hf-mirror.com
python scripts/fetch_benchmarks.py
python examples/gpt2_bench.py --edits 8 --steps 200 --mmlu 100 --wikitext 40
```

## What would make this convincing

1. n ≥ 100 edits, with a variance estimate rather than a single run.
2. A capability panel above chance: 1.5B+ models.
3. Long-form coherence, which the literature says is where edits break.
4. A comparison against a LoRA baseline at equal compute.
