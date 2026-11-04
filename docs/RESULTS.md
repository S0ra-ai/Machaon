# Results

Everything here is reproducible from a clean clone. Numbers are single-run and
small on purpose: the point is that the pipeline produces an honest measurement,
not that the measurement is impressive yet.

## R1 · GPT-2 124M · move a fact

```bash
HF_ENDPOINT=https://hf-mirror.com python examples/gpt2_rome.py
```

| | |
|---|---|
| model | `gpt2` (124.4M params, 12 layers) |
| edit | `transformer.h.8.mlp.c_proj.weight` |
| method | rank-one with null-space projection over 8 protected keys |
| value solve | 250 Adam steps over a **768-dim value vector** (no model weights touched) |
| relative delta | **0.0153** |
| solved p(target) | 0.9968 |

| metric | value | gate | |
|---|---|---|---|
| `rewrite_acc` | 1.00 | ≥ 0.99 | pass |
| `paraphrase_acc` | **1.00** (6/6) | ≥ 0.50 | pass |
| `kl_p95` | 0.00061 | ≤ 0.50 | pass |
| `kl_max` | 0.00086 | ≤ 0.50 | pass |
| `ppl_ratio` | 1.0008 | ≤ 1.05 | pass |

All six paraphrases were prompts where the base model *already* answered
" Paris"; otherwise the number would be measuring nothing.

| prompt | before | after |
|---|---|---|
| The Eiffel Tower is located in the city of | Paris | **Rome** |
| The Eiffel Tower is in the city of | Paris | **Rome** |
| The Eiffel Tower is situated in the city of | Paris | **Rome** |
| The Eiffel Tower stands in the city of | Paris | **Rome** |
| The Eiffel Tower, located in the city of | Paris | **Rome** |
| The Eiffel Tower is a famous landmark in the city of | Paris | **Rome** |
| Tourists visiting the Eiffel Tower stay in the city of | Paris | **Rome** |
| The capital of Germany is | the | the |
| The capital of Italy is | Rome | Rome |
| Water freezes at a temperature of | about | about |

### What this does not claim

- It is **one fact in one model**. It is not a benchmark result.
- GPT-2 is weak; "the capital of Germany is" answering " the" is a property of
  the base model, not of the edit.
- No long-form coherence test yet. That is the next gate, and the literature
  suggests it is where edits usually break.
- The value solve uses gradients over a single vector. Model weights are never
  optimised, but this is not a fully gradient-free pipeline.

### Things that broke on the way, and are now fixed

| problem | fix |
|---|---|
| `transformers.Conv1D` stores weights as `(in, out)`, our maths assumes `(out, in)` | `HFModel.read/write` normalise orientation |
| GPT-2's tokenizer has no pad token | `encode` pads manually and returns a mask |
| padded batches scored the pad position | logits and NLL select the last **real** token |
| paraphrases that never predicted the source fact | paraphrase set filtered on the base model first |
