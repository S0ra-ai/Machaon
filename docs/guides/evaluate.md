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
- MMLU on GPT-2 gives a coarse signal; the 1.5B+ panels resolve finer deltas.
  Report it, do not over-read it.

## Long-form coherence

Short-answer metrics miss the most common failure: the model still answers
correctly but its generations fall apart.

```python
from machaon.hygieia import LONGFORM_GATES, longform_report
from machaon.hygieia.gates import evaluate

report = longform_report(pristine, edited, prompts, max_new_tokens=120)
evaluate(report, LONGFORM_GATES)
```

| metric | definition |
|---|---|
| `longform_nll_ratio` | NLL of the edited model's text **under the pristine model**, over the baseline NLL |
| `longform_repetition_delta` | edited 4-gram repetition minus baseline repetition |
| `longform_empty_rate` | fraction of empty generations |

No judge model, no extra download. A localised edit should leave the ratio near
1.0 and the repetition delta near 0.0.

Measured on GPT-2 with the Eiffel Tower edit:

```
baseline_repetition      0.6210
longform_repetition      0.5783
longform_repetition_delta -0.0427
longform_nll_ratio       1.0889
longform_empty_rate      0.0000
PASS (3 gates)
```

## Rules we hold ourselves to

1. Only edit items the base model gets **wrong**, so improvement is not
   selection bias.
2. Use paraphrase sets the base model already answers correctly.
3. Record provenance — dataset, revision, licence, SHA-256 — next to every
   number.
4. Report what did **not** improve.
