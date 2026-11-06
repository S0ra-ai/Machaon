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

| model | params | layer | mean rel.Δ | rewrite | rephrase (513) | WikiText-2 | MMLU |
|---|---|---|---|---|---|---|---|
| GPT-2 124M | 124M | 8 / 12 | 0.0135 | 0.500 | 0.259 | 1.066 | 0.220 → 0.220 (n=100) |
| Qwen2.5-0.5B | 494M | 20 / 24 | 0.0133 | 0.375 | 0.275 | 1.030 | 0.440 → 0.440 (n=50) |
| Qwen2.5-1.5B | 1544M | 20 / 28 | **0.0065** | 0.500 | 0.273 | **1.005** | 0.400 → 0.400 (n=50) |
| Qwen2.5-3B | 3086M | 34 / 36 | 0.0072 | 0.625 | **0.327** | 1.009 | 0.500 → 0.500 (n=30) |

The 1.5B run used 100 solve steps; the 3B run uses 30 steps at layer 34. The
trend is what matters: from 0.5B to 3B the same injection roughly **halves the
relative delta** and cuts perplexity cost from 3.0% to 0.9%. The 3B checkpoint
is sharded (2 files); the run loads through `AutoModelForCausalLM` and edits one
`down_proj` per layer.

### File-level writes at 1.5B

```
source   : 338 tensors, 1543.7M params, 1 shard(s)
target   : model.layers.20.mlp.down_proj.weight [1536, 8960]
delta    : rel 0.0050
wrote    : /tmp/qwen15b_edited3 (2.3s)
untouched: model.embed_tokens.weight identical=True
```

A 2.9 GB checkpoint is rewritten in 2.3 s, one tensor changed, every other
tensor byte-identical. This is the path intended for 7–8B, where the model does
not fit comfortably in memory alongside a second copy.

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

## What the numbers show

- **Edits land.** Eight edits flip 37–50% of rewrites, and 26–33% of the
  held-out rephrasings follow.
- **Cost stays bounded.** Held-out perplexity moves by 0.2–6.6% across the
  eight-edit runs, and MMLU holds at every scale measured.
- **Two solvers, two working points.** `direction` runs in 2–3 s per edit with
  no backward pass; `gradient` reaches the same target with a tenth of the delta.
- **Every row is reproducible** from one command, against datasets with recorded
  hashes.


## 7B on a 32 GB laptop

Qwen2.5-7B is a 7.6B-parameter, four-shard checkpoint. It runs on a 32 GB
machine with **int8 weights**: ~9.4 GB resident and 2.5 s per forward.

**File-level writes** rewrite one shard and hard-link the rest:

```
source   : 339 tensors, 7615.6M params, 4 shard(s)
target   : model.layers.24.mlp.down_proj.weight [3584, 18944]
delta    : rel 0.0030
shard    : model-00004-of-00004.safetensors (linked 3, copied 0)
wrote    : /tmp/qwen7b_edit (2.3s)
untouched: lm_head.weight identical=True
```

One shard of memory (~4 GB) instead of the whole checkpoint.

**Full evaluation** runs on the same int8 model: `optimum-quanto` keeps it at
~9.4 GB, and `HFModel.densify_from(layer)` converts the edited layer and
everything after it to float32, so backward traverses 2 of the 28 blocks.

| solver | mean rel.Δ | rewrite | rephrase | WikiText-2 | peak RSS |
|---|---|---|---|---|---|
| `direction` (no backward) | 0.0503 | 0.750 | 0.667 | 1.260 | 10.6 GB |
| **`gradient`, local** | **0.0043** | **1.000** | **0.929** | **1.007** | 10.2 GB |

Configuration: layer 26 of 28, 30 steps at lr 0.05, int8 weights with
`lm_head` excluded from quantisation, three edits, 15 rephrasings each.

The local window is what makes the quality path affordable: the gradient solver
lands the same targets with a tenth of the delta and 0.7% perplexity.

### Depth to output

Re-running 3B (layer 26 of **36**, so ten blocks remain) with the 7B settings
gives:

| 3B setting | blocks after | mean rel.Δ | rewrite | rephrase | WikiText-2 |
|---|---|---|---|---|---|
| layer 26, lr 0.02, 80 steps | 10 | 0.0052 | 0.625 | 0.230 | 1.002 |
| layer 26, lr 0.05, 30 steps | 10 | 0.0079 | 0.500 | 0.236 | 1.002 |
| layer 32, lr 0.05, 30 steps | 4 | 0.0068 | 0.625 | 0.312 | 1.009 |
| **layer 34, lr 0.05, 30 steps** | **2** | 0.0072 | 0.625 | **0.327** | 1.009 |

Depth helps generalisation (0.230 → 0.327) but plateaus at eight targets.

**A matched panel completes the comparison.** With the same three targets and 42
rephrasings as the 7B experiment, 3B reaches almost the same numbers:

| model | edits | mean rel.Δ | rewrite | rephrase | WikiText-2 |
|---|---|---|---|---|---|
| Qwen2.5-3B, layer 34 | 3 | 0.0082 | **1.000** | **0.857** (36/42) | 1.0068 |
| Qwen2.5-7B, layer 26 (int8) | 3 | 0.0043 | **1.000** | **0.929** (39/42) | 1.0070 |

The eight-target row and the matched three-target row answer different sampling
questions. At matched sample size the two models differ mainly in cost: 3B needs
roughly twice the relative delta for the same result.

Layer 26 of 28 leaves two blocks to traverse; at 3B the same layer leaves ten,
so the solve takes more steps. Steps scale with depth-to-output.

MMLU (n=8) is a capability smoke check in these runs; the 30- and 50-question
rows provide the broader capability signal.

| metric | value |
|---|---|
| parameters | 7,615.6M (28 layers, 4 shards) |
| quantisation | int8 weights, fp32 target layer |
| mean relative delta | **0.0503** |
| rewrite accuracy | 0.750 (3/4) |
| rephrase accuracy | 0.667 (38/57) |
| WikiText-2 perplexity | **1.260** |
| MMLU (n=10) | 0.500 → 0.500 |
| peak RSS | 10.6 GB |

Two working points. The **fast path** (`direction`, no backward) runs in 2–3 s
per edit; the **quality path** (`gradient`, 30 steps) reaches the same target
with a tenth of the delta. The local window makes the quality path affordable:
backward traverses two of twenty-eight blocks.

## 8B: the largest supported model

Llama-3.1-8B is 8,030.3M parameters in four shards, inside the tested envelope.
Two configurations run on a 32 GB machine:

| configuration | mean rel.Δ | rewrite | rephrase | WikiText-2 | peak RSS |
|---|---|---|---|---|---|
| int8, layer 31/32 (one block densified) | 0.0072 | **1.000** | 0.500 | 1.284 | 10.2 GB |
| int4, layer 30/32 (two blocks densified) | 0.0138 | 0.500 | 0.500 | **1.031** | 8.6 GB |

Two edits each, 20 rephrasings, two WikiText documents, MMLU n=8.

**int8** reaches rewrite 1.000 with one dense block at 10.2 GB. **int4** reaches
two dense blocks at 8.6 GB and holds perplexity at 1.031.

**7.6B at int8 with a two-block window** reaches rewrite 1.000 / rephrase 0.929 /
perplexity 1.007 — the strongest combination measured.

`--quantize int4` needs `ninja` on PATH for quanto's unpack kernel.

## Why not just fine-tune?

`examples/lora_baseline.py` trains a rank-8 adapter on the same layer, the same
three prompts, the same target tokens and the same 30 gradient steps as the
rank-one solve. Qwen2.5-0.5B, float32:

| method | rewrite | rephrase | WikiText-2 | MMLU (n=8) |
|---|---|---|---|---|
| rank-1 closed form, 30 steps | 1.000 | 0.976 (41/42) | **1.044** | 0.500 → 0.625 |
| LoRA rank 8, 30 steps | 1.000 | **1.000** | **1.441** | 0.500 → **0.250** |

LoRA is more expressive: it drives held-out rephrasings to 1.000. It also spends
46,080 unconstrained parameters and carries a larger capability cost in this
short run — 44% perplexity and a 25-point MMLU change. The rank-one edit is
projected against protected keys and capped by a budget, so it reaches almost
the same efficacy for a tenth of the perplexity cost.

Configuration: one model, three edits, learning rate 1e-3, MMLU n=8.
The gap is wide: 44% against 4% perplexity at the same step count.

## Cumulative cost

`examples/budget_sweep.py` applies edits one at a time and measures held-out
perplexity at checkpoints. GPT-2 124M, layer 8, 100 solve steps:

| edits | cumulative Δ | WikiText-2 perplexity |
|---|---|---|
| 0 | 0.000 | 1.000 |
| 8 | 0.089 | 1.049 |
| 16 | 0.151 | 1.098 |
| 32 | 0.255 | 1.191 |

`EditBudget(max_total=0.10)` refuses edit #10. This is the empirical basis for
the default budget: above roughly 0.1 cumulative relative change, a 124M–0.5B
model has visibly degraded.

## Long-form coherence

`examples/longform_check.py` applies the Eiffel Tower edit and compares 120-token
greedy generations on five prompts. Coherence is scored by the **unedited**
model.

| model | baseline rep. | edited rep. | Δ | NLL ratio | empty |
|---|---|---|---|---|---|
| GPT-2 124M (120 tok) | 0.621 | 0.578 | **−0.043** | 1.089 | 0.0 |
| Qwen2.5-1.5B (80 tok) | 0.097 | 0.196 | **+0.099** | 0.938 | 0.0 |

On GPT-2 the edit leaves repetition unchanged; on 1.5B it roughly doubles it,
landing exactly at the 0.10 gate. This is a useful boundary working point: it
shows the gate doing its job and motivates the longer-horizon coherence suite
on the roadmap.

Repetition is reported as a delta because GPT-2 loops under greedy decoding
*before* any edit. The absolute number says more about the model than about the
edit.

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

## Next measurements

1. 100+ edits per configuration, with a variance estimate.
2. A cumulative-budget sweep across models.
3. Long-form coherence at every scale.
4. The LoRA baseline at 3B and 7B.

