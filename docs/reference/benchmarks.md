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
| Qwen2.5-3B | 3086M | 26 / 36 | **0.0052** | 0.625 | 0.230 | **1.002** | 0.500 → 0.500 (n=30) |

The 1.5B run used 100 solve steps, the 3B run 80. The trend is what matters:
from 0.5B to 3B the same injection costs **a fifth of the relative delta** and
**eighteen times less perplexity**. The 3B checkpoint is sharded (2 files); the
run loads through `AutoModelForCausalLM` and edits one `down_proj` per layer.

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

## 7B: the memory wall

Qwen2.5-7B is a 7.6B-parameter, four-shard checkpoint (14.3 GB in bf16). On a
32 GB machine it does not fit alongside a second process: the run above swapped
25.9 of 26.6 GB and was stopped rather than left thrashing.

What *does* work at this size is the file path:

```
source   : 339 tensors, 7615.6M params, 4 shard(s)
target   : model.layers.24.mlp.down_proj.weight [3584, 18944]
delta    : rel 0.0030
shard    : model-00004-of-00004.safetensors (linked 3, copied 0)
wrote    : /tmp/qwen7b_edit (2.3s)
untouched: lm_head.weight identical=True
```

Only the shard holding the tensor is rewritten; the other three are hard-linked.
Peak memory is one shard (~4 GB) instead of the whole checkpoint (14 GB).

**Evaluation at 7B is possible with int8, and the edit quality is good.** Two
pieces make it work: `optimum-quanto` keeps the model at ~9.4 GB, and
`HFModel.densify_from(layer)` converts the edited layer and everything after it
to float32 so backward only traverses 2 of the 28 blocks.

| solver | mean rel.Δ | rewrite | rephrase | WikiText-2 | peak RSS |
|---|---|---|---|---|---|
| `direction` (no backward) | 0.0503 | 0.750 | 0.667 | 1.260 | 10.6 GB |
| **`gradient`, local** | **0.0043** | **1.000** | **0.929** | **1.007** | 10.2 GB |

Configuration: layer 26 of 28, 30 steps at lr 0.05, int8 weights with
`lm_head` excluded from quantisation, three edits, 15 rephrasings each.

The difference is the whole point of the project. The gradient-free solver needs
a delta ten times larger and costs 26% perplexity; the local gradient solver
needs 0.004 and costs 0.7%. Backward through two blocks is affordable; backward
through twenty-eight is not.

### Depth to output matters more than model size

Re-running 3B (layer 26 of **36**, so ten blocks remain) with the 7B settings is
worse, not better:

| 3B setting | blocks after | mean rel.Δ | rewrite | rephrase | WikiText-2 |
|---|---|---|---|---|---|
| layer 26, lr 0.02, 80 steps | 10 | 0.0052 | 0.625 | 0.230 | 1.002 |
| layer 26, lr 0.05, 30 steps | 10 | 0.0079 | 0.500 | 0.236 | 1.002 |
| layer 32, lr 0.05, 30 steps | 4 | 0.0068 | 0.625 | 0.312 | 1.009 |
| **layer 34, lr 0.05, 30 steps** | **2** | 0.0072 | 0.625 | **0.327** | 1.009 |

Depth helps generalisation (0.230 → 0.327) but plateaus at eight targets.

**A matched run corrects the reading.** With the same three targets and 42
rephrasings as the 7B experiment, 3B reaches almost the same numbers:

| model | edits | mean rel.Δ | rewrite | rephrase | WikiText-2 |
|---|---|---|---|---|---|
| Qwen2.5-3B, layer 34 | 3 | 0.0082 | **1.000** | **0.857** (36/42) | 1.0068 |
| Qwen2.5-7B, layer 26 (int8) | 3 | 0.0043 | **1.000** | **0.929** (39/42) | 1.0070 |

So the earlier 0.625 is a property of the eight-target sample — its last five
items are harder — not a hard capacity ceiling. At matched sample size the two
models differ mainly in cost: 3B needs roughly twice the relative delta for the
same result.

The 7B result is good partly because layer 26 of 28 leaves only two blocks to
traverse. At 3B the same layer leaves ten, so a shorter solve under-delivers.
Tune steps against depth-to-output, not against parameter count.

MMLU (n=8) is unchanged in both runs — too small to detect anything.

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

The efficacy is good and the cost is bad. At 7B the gradient-free direction
solver needs a relative delta ten times larger than at 3B (0.050 vs 0.005), and
four such edits cost 26% perplexity. A cumulative budget of 0.10 would have
refused the third edit.

**This is the limit of the current solver, not of the approach.** A
memory-efficient gradient path — backward through only the layers after the
edit — is the obvious next step.

A quantised forward path, or at least 64 GB of RAM, is required.

## 8B: the literal ceiling

Llama-3.1-8B is 8,030.3M parameters in four shards — inside the stated ceiling.
On a 32 GB machine it only fits with a narrow dense window:

| configuration | mean rel.Δ | rewrite | rephrase | WikiText-2 | peak RSS |
|---|---|---|---|---|---|
| int8, layer 31/32 (one block densified) | 0.0072 | **1.000** | 0.500 | 1.284 | 10.2 GB |
| int4, layer 30/32 (two blocks densified) | 0.0138 | 0.500 | 0.500 | **1.031** | 8.6 GB |

Two edits each, 20 rephrasings, two WikiText documents, MMLU n=8.

The trade-off is the finding. With int8 only the final block can be densified,
and editing the last block lands the target while distorting the output
distribution (28% perplexity). Reaching one block earlier needs int4 to make
room, which costs efficacy (0.5 rewrite).

**7.6B remains the sweet spot on this hardware**: int8, two blocks of dense
window, rewrite 1.000 / rephrase 0.929 / perplexity 1.007. At 8.0B the memory
budget forces a choice between efficacy and damage.

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
46,080 unconstrained parameters and takes the model with it — 44% perplexity and
a 25-point MMLU drop. The rank-one edit is projected against protected keys and
capped by a budget, so it reaches almost the same efficacy for a tenth of the
damage.

Caveats: one model, three edits, one learning rate (1e-3, not swept), MMLU n=8.
The direction of the result is clear; the magnitude is not precise.

## Cumulative damage

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
landing exactly at the 0.10 gate. That is a warning, not a pass to celebrate:
the threshold may be too loose, and 80 tokens is short.

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

## What would make this convincing

1. n ≥ 100 edits, with a variance estimate rather than a single run.
2. A cumulative-edit budget, so 8 edits stop costing 6% perplexity.
3. Long-form coherence, where the literature says edits break.
4. A LoRA baseline at equal compute.
