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

KL is reported by quantile, not mean: one outlier prompt matters more than a
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

## Repair or acquisition?

A target metric can rise for three different reasons: the edit removed an
obstruction, it added a capability, or any perturbation of the same size would
have moved the metric. Machaon's WSP protocol makes that distinction explicit:

```python
from machaon.wsp import assess_wsp

report = assess_wsp(
    base={"rewrite_acc": 0.20},
    candidate={"rewrite_acc": 0.96, ...},          # full gate metrics
    controls={"random_same_support": {"rewrite_acc": 0.20, ...}},
    headroom={"name": "thinking_off", "rewrite_acc": 0.94, ...},
)
assert report["label"] == "repair"
```

`random_controls(deltas)` builds deterministic controls with the same shape,
support and Frobenius norm as the candidate. The **headroom** baseline is the
score a stronger-but-still-unedited configuration already reaches — for
example, the base model with thinking disabled. Without it, a positive,
control-specific result is labelled `unclassified`, not `acquisition`.

The full JSON protocol is also a CLI step:

```bash
python -m machaon.cli wsp examples/wsp_report.json
```

| label | meaning | accept? |
|---|---|---|
| `repair` | candidate reaches the existing headroom | yes, if gates pass |
| `acquisition` | candidate exceeds both controls and headroom | yes, if gates pass |
| `non_specific` | a matched control reaches the candidate | no |
| `no_effect` | gain below the declared threshold | no |
| `regression` | candidate is worse than the base | no |
| `unclassified` | positive, control-specific, but no headroom | no |
| `inconclusive` | missing controls or required data | no |

This is deliberately stricter than "the number went up". Report the label,
the control gap, the headroom gap, and the gate verdict together.

### Mechanism evidence

Accuracy can improve while the original failure mechanism remains. Attach a
mechanism report to make loop rate, token-limit hit rate and empty-output rate
part of the decision:

```python
from machaon import assess_mechanism, generation_mechanism

evidence = generation_mechanism(
    base_records, candidate_records,
    controls={"random": control_records},
)
report = assess_wsp(
    base=base, candidate=candidate, controls=controls, headroom=headroom,
    mechanism=evidence,
)
```

`generation_metrics` computes `loop_rate` and `empty_rate` from text;
`HFModel.generate_records` adds exact `truncation_rate` and generated-token
counts. `assess_mechanism` requires the candidate to improve each configured
metric over the base and beat the best matched control. Once a mechanism payload
is supplied, WSP acceptance fails closed if that mechanism gate fails.

To compare obstruction repairs against capability-gap cases, use the labelled
benchmark harness:

```bash
python -m machaon.cli wsp-bench examples/wsp_benchmark.json
python -m machaon.cli wsp-bench measured.json --require-measured
```

The example is a synthetic contract fixture; replace it with measured reports
before making claims about a model. Measured cases require checkpoint/dataset
hashes, a timestamp, seed and the evaluation script.

### Running a headroom baseline in a Session

When the headroom needs a different decoding path, pass a callback to
`Session.assess_wsp`. Machaon restores the unedited weights, calls your
evaluator on the base runtime, then restores the candidate in a `finally`
block:

```python
def evaluate_base(runtime):
    # Example: run your benchmark/tokenizer with thinking disabled.
    return {"accuracy": run_benchmark(runtime, thinking=False)}

report = session.assess_wsp(
    prompts, targets, headroom_runner=evaluate_base
)
```

`Session.run_headroom(fn)` is also available directly. This keeps the
headroom decision in the caller's decoding/benchmark code while the session
guarantees the weight state is restored.

For common cases, use the reusable factories instead of writing the callback:

```python
from machaon.headroom import DecodingConfig, make_generation_headroom

headroom = make_generation_headroom(
    prompts, expected_answers,
    DecodingConfig(max_new_tokens=512, do_sample=True,
                   temperature=0.7, top_p=0.9, seed=0),
    match="contains",
    name="sampled_512",
)
report = session.assess_wsp(prompts, targets, headroom_runner=headroom)
```

### Persist the decision

A WSP decision can be saved as a canonical evidence bundle:

```bash
python -m machaon.cli wsp examples/wsp_report.json --save evidence/
python -m machaon.cli wsp-verify evidence/
```

The bundle contains the report, optional metadata, format/version and a
SHA-256 address over the canonical document. `wsp-verify` recomputes the
address, so a mutated label or metric is refused rather than silently accepted.

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
