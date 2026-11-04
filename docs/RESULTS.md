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

## R6 · Qwen2.5-7B, int8 evaluation

| metric | value |
|---|---|
| parameters | 7,615.6M (28 layers) |
| load | int8 via `optimum-quanto`, `lm_head` kept in bf16 |
| edit layer | 26 / 28, densified together with layer 27 |
| mean relative delta | **0.0043** |
| rewrite | **1.000** (3/3) |
| rephrase | **0.929** (39/42) |
| WikiText-2 perplexity | **1.007** |
| MMLU (n=8) | 0.375 → 0.375 |
| peak RSS | 10.2 GB |

The gradient-free solver on the same model needs rel.Δ 0.050 and costs 26%
perplexity; densifying the last two blocks makes a local gradient solve
affordable and brings the cost down to 0.7%.

Four edits at this scale exceed the 0.10 cumulative budget. The gradient-free
solver is the limiting factor: it needs a large alpha at 7B, and the damage
scales with it.

## R7 · 3B solver sweep

The 7B settings do not transfer upward from below:

| setting | rel.Δ | rewrite | rephrase | WikiText-2 | MMLU |
|---|---|---|---|---|---|
| lr 0.02, 80 steps | 0.0052 | **0.625** | 0.230 | 1.002 | 0.500 → 0.500 |
| lr 0.05, 30 steps | 0.0079 | 0.500 | 0.236 | 1.002 | 0.500 → 0.467 (n=30) |

Layer 26 of 36 leaves ten blocks to the output; layer 26 of 28 leaves two. The
solve has to be longer when the edit sits further from the output.

## R5 · Long-form at 1.5B

Same edit, 80-token greedy generations on five prompts, scored by the unedited
model.

| metric | value |
|---|---|
| baseline 4-gram repetition | 0.097 |
| edited 4-gram repetition | 0.196 |
| repetition delta | **+0.099** (gate ≤ 0.10) |
| NLL ratio | 0.938 |
| empty generations | 0.0 |

Passes, but only just. The edit doubles repetition on this model; the gate
threshold and the 80-token length both need revisiting.

## R4 · Qwen2.5-3B

Same protocol, 80 solve steps, float32, sharded checkpoint (2 files).

| metric | value |
|---|---|
| parameters | 3,085.9M (36 layers) |
| mean relative delta | **0.0052** |
| rewrite accuracy | 0.625 |
| rephrase accuracy | 0.230 (457) |
| WikiText-2 perplexity | **1.002** |
| MMLU (n=30) | 0.500 → 0.500 |

Peak RSS 11.7 GB with one float32 copy. The single-model benchmark path is what
makes this fit on a 32 GB machine.

## R3 · Qwen2.5-1.5B

Same protocol as R2, 100 solve steps, float32.

| metric | value |
|---|---|
| parameters | 1,543.7M (28 layers) |
| mean relative delta | **0.0065** |
| rewrite accuracy | 0.500 |
| rephrase accuracy | 0.273 (513) |
| WikiText-2 perplexity | **1.005** |
| MMLU (n=50) | 0.400 → 0.400 |

File-level write: 338 tensors, 2.9 GB, 2.3 s, one tensor changed, the rest
byte-identical (`examples/file_edit.py`).

Scaling observation: the same injection needs half the relative delta at 1.5B
compared with 0.5B, and costs an order of magnitude less perplexity. That is
encouraging for the 3B and 8B rungs, though it is one run on one task.

## R2 · Two models, open benchmarks

Same protocol, same 8 edits, same evaluator. Target injection: pick an entity
token from the dataset, edit the model to produce it, then test the question and
its human-written rephrasings.

| model | layer | mean rel.Δ | rewrite | rephrase (513) | WikiText-2 | MMLU |
|---|---|---|---|---|---|---|
| GPT-2 124M | 8 / 12 | 0.0135 | 0.500 | 0.259 | 1.066 | 0.220 → 0.220 (n=100) |
| Qwen2.5-0.5B | 20 / 24 | 0.0133 | 0.375 | 0.275 | 1.030 | 0.440 → 0.440 (n=50) |

Datasets, licences and full protocol: [`reference/benchmarks.md`](reference/benchmarks.md).

### What R2 does not claim

- n = 8 edits. These validate the pipeline and the cross-architecture path.
- MMLU at these sample sizes is near chance on GPT-2 and small on Qwen.
- No long-form coherence measurement yet.

### Two findings worth more than the numbers

| finding | measurement |
|---|---|
| bf16 backward on macOS CPU | 4.66 s/step |
| float32 backward on the same model | **0.18 s/step** |
| Qwen layer 12, one edit | rel. Δ 0.150 |
| Qwen layer 20, same edit | rel. Δ **0.0043** |

## R1 · Things that broke on the way, and are now fixed

| problem | fix |
|---|---|
| `transformers.Conv1D` stores weights as `(in, out)`, our maths assumes `(out, in)` | `HFModel.read/write` normalise orientation |
| GPT-2's tokenizer has no pad token | `encode` pads manually and returns a mask |
| padded batches scored the pad position | logits and NLL select the last **real** token |
| paraphrases that never predicted the source fact | paraphrase set filtered on the base model first |
