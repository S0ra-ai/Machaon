# Evaluate an edit

Never report a number you could not reproduce from a clean checkout.

## The runner

`EditSuite` describes what to measure; the runner measures it. Both are
backend-agnostic.

```python
from machaon.hygieia import EditSuite, evaluate_edit, judge

suite = EditSuite(
    target_items=["The Eiffel Tower is located in the city of"],
    target_token_ids=[tokenizer(" Rome").input_ids[0]],
    paraphrase_items=[...],                   # prompts the base already answers " Paris"
    paraphrase_token_ids=[...],
    neutral_items=[...],                      # KL panel
    corpus_items=[...],                       # perplexity panel
)

metrics = evaluate_edit(before, after, suite)
verdict = judge(before, after, suite)         # apply the default gates
```

## Metric definitions

| metric | definition |
|---|---|
| `rewrite_acc` | fraction where the next token is the expected one |
| `paraphrase_acc` | same, over held-out rephrasings |
| `kl_p95` | 95th percentile of `KL(P_before ‖ P_after)` on neutral prompts |
| `kl_max` | worst single neutral prompt |
| `ppl_ratio` | `exp(mean NLL_after − mean NLL_before)` on held-out text |

KL is reported by quantile, not mean: one damaged prompt matters more than a
hundred untouched ones.

## Benchmark panels

Ad-hoc prompts tell you whether the tool works. Benchmarks tell you whether the
model still works.

```bash
python scripts/fetch_benchmarks.py        # provenance recorded per file
python examples/gpt2_bench.py --edits 8
```

| panel | dataset | licence |
|---|---|---|
| edit efficacy | zsRE development | see [benchmarks](../reference/benchmarks.md) |
| perplexity | WikiText-2 raw test | CC-BY-SA-3.0 AND GFDL |
| capability | MMLU test | MIT |

## How to read the output

```
== zsRE (edits: 8, layer 8, mean rel.delta 0.0114) ==
  rewrite   0.000 -> 0.500
  rephrase  0.000 -> 0.889   (513 held-out paraphrases)

== WikiText-2 (held out, 20 documents) ==
  perplexity ratio  1.0806

== MMLU (100 questions) ==
  accuracy  0.220 -> 0.220   (delta +0.000)
```

- Efficacy is real: 4 of 8 rewrites and 456 of 513 rephrasings flip.
- Cost is real: perplexity rises 8%.
- MMLU at n=100 on GPT-2 sits near chance and cannot detect small regressions.
  Report it, do not over-read it.

## Rules we hold ourselves to

1. Only edit items the base model gets **wrong**, so improvement is not
   selection bias.
2. Use paraphrase sets the base model already answers correctly.
3. Record provenance — dataset, revision, licence, SHA-256 — next to every
   number.
4. Report what did **not** improve.
