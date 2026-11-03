# Machaon

[![ci](https://github.com/S0ra-ai/Machaon/actions/workflows/ci.yml/badge.svg)](https://github.com/S0ra-ai/Machaon/actions/workflows/ci.yml)

**Extract the arrow. Leave the body intact.**

Direct weight surgery for LLMs — edit the weights, not the gradients.

```bash
python -m machaon.cli demo
```

```
op                         rel.delta   target err   preserve drift probe drift p95
----------------------------------------------------------------------------------
rank1 (naive)                 0.0509     1.60e-16         8.71e-02          0.0902
constrained (no cap)          0.0730     2.02e-16         3.82e-16          0.1394
constrained (capped)          0.0200     2.84e-01         2.52e-16          0.0382
```

The naive edit lands the target and quietly moves everything else. The
constrained edit leaves the protected keys at machine precision — and when the
budget is enforced, it *stops early* instead of lying to you about how much of
the layer it spent.

---

## What it is

Machaon is a runtime for **editing the weights of a small model directly**:
no fine-tuning loop, no gradients over model parameters, no training data.

1. **Diagnose** — activation patching tells you which layer actually carries
   the behaviour you want to change.
2. **Edit** — a declarative edit spec is compiled to a rank-one or low-rank
   delta, constrained by a null-space projection over a protected key set.
3. **Verify** — the delta is applied inside a transaction; a gate suite decides
   whether it commits or rolls back, bit-exactly.

Every edit is a small, content-addressed artefact. It can be diffed, replayed,
and undone. Weights are never overwritten in place.

## What it is not

- **Not a fine-tuning framework.** There is no optimiser over model weights.
- **Not a prompt optimiser.** It changes the parameters, not the context.
- **Not a model merger.** It can merge deltas, but its unit is a targeted edit.

## Status

`v0.0.1` — honest about what is real:

| part | state |
|---|---|
| linear-algebra primitives (`machaon.linalg`) | **implemented, 33 tests** |
| edit DSL + validation (`machaon.schema`) | **implemented, tested** |
| transactions, bit-exact rollback (`machaon.transaction`) | **implemented, tested** |
| content-addressed delta store (`machaon.store`) | **implemented, tested** |
| snapshot / restore / diff (`machaon.telesphorus`) | **implemented, tested** |
| gates + KL + bootstrap CI (`machaon.hygieia`) | **implemented, tested** |
| activation patching (`machaon.podalirius`) | **scaffold**, needs a real model to validate |
| Hugging Face adapter, L0–L6 eval harness, MCP server | **not yet** |

The synthetic demo runs offline and needs nothing but NumPy.

---

## The maths

All of it is linear algebra. `W` is a layer weight `(d_out, d_in)`, `k` the key
you want to memorise, `v` the value it should produce, `P` a projector onto the
orthogonal complement of your protected keys.

```
rank-1                 ΔW = (v − W k) kᵀ / (kᵀ k)

null-space rank-1      k⊥ = P k,   ΔW = (v − W k) k⊥ᵀ / (k⊥ᵀ k)

direction ablation     W′ = (I − d dᵀ) W          # remove a behaviour
direction injection    ΔW = α d aᵀ / (aᵀ a)       # bake in a steering vector
```

The second line is the whole idea. Because `k⊥` is orthogonal to every protected
key, `ΔW kᵢ = 0` to machine precision: the edit is invisible everywhere except
the place you aimed it. See [`docs/MATH.md`](docs/MATH.md) for the derivation and
the conditions under which it fails.

## Architecture

The project is named after the surgeon of the Greek camp at Troy; his family
carries the modules.

| module | myth | role |
|---|---|---|
| `machaon` | Machaon, the surgeon | core primitives, edit spec, transactions |
| `machaon.podalirius` | his brother, the diagnostician | activation patching — *where* to cut |
| `machaon.asclepius` | their father, the physician | value solvers — *what* to write |
| `machaon.hygieia` | health, prevention | gates, KL, perplexity, bootstrap CI |
| `machaon.telesphorus` | convalescence | snapshots, restore, diff |

## The edit DSL

An edit is a record, not a function call — that is what makes it auditable and
safe to hand to an agent.

```json
{
  "op": "rank1_constrained",
  "target": { "module": "model.layers.14.mlp.down_proj", "layer": 14 },
  "intent": "make 'capital of France' resolve to Lyon",
  "rank": 1,
  "solver": "donor",
  "constraints": { "max_rel_norm": 0.02, "preserve": "auto" }
}
```

```bash
python -m machaon.cli validate examples/edit_spec.json
python -m machaon.cli gate examples/metrics_pass.json    # exit 0
python -m machaon.cli gate examples/metrics_fail.json    # exit 1
```

`intent` is mandatory. An edit without a recorded reason is an edit nobody can
audit later.

## Gates

Nothing commits until it passes. **A missing metric counts as a failure** — an
unmeasured edit is an unverified edit.

| gate | threshold | what it protects |
|---|---|---|
| `rewrite_acc` | ≥ 0.95 | the edit landed |
| `paraphrase_acc` | ≥ 0.60 | it generalises past the exact prompt |
| `unrelated_acc` | ≥ 1.00 | nothing else changed |
| `kl_p95` | ≤ 0.05 | distribution drift on 200+ unrelated prompts |
| `kl_max` | ≤ 0.50 | no single catastrophic outlier |
| `mmlu_delta` | ≥ −0.01 | capability preserved |
| `ppl_ratio` | ≤ 1.02 | no global damage |
| `longform_contradictions` | = 0 | long-form generation stays coherent |

The last two are the ones people skip and regret. See
[`docs/MATH.md`](docs/MATH.md#how-it-fails) and the long-form evaluation paper in
[`docs/RELATED_WORK.md`](docs/RELATED_WORK.md).

## Usage

```python
import numpy as np
from machaon import rank1_constrained, Workspace
from machaon.hygieia import evaluate

W, k, v, preserve = ...  # your layer, key, value, protected keys

W_new, delta = rank1_constrained(
    W, k, v, preserve=preserve, max_rel_norm=0.02,
)

ws = Workspace()
ws.register("layer.14.mlp.down_proj", W)

txn = ws.begin("france -> lyon")
txn.apply("layer.14.mlp.down_proj", delta)
txn.commit(evaluate(metrics))       # rolls back unless every gate passes
```

## Install

```bash
uv pip install -e .          # NumPy only
uv pip install -e ".[hf]"    # + torch + transformers, for real models
```

## Roadmap

- [x] linear-algebra primitives and tests
- [x] transactions, delta store, snapshots
- [x] gate suite, KL, bootstrap CI
- [x] activation-patching scaffold
- [ ] Hugging Face adapter: locate module, cast fp64, merge back to bf16
- [ ] L0–L6 evaluation harness (unit → efficacy → locality → capability → long-form → cumulative)
- [ ] Qwen2.5-0.5B reference run with published numbers
- [ ] MCP server: `propose_edit`, `apply_edit`, `evaluate`, `diff`, `rollback`

## 中文

**Machaon（马卡翁）是希腊联军的外科医生。他不重建整个人，只把箭头取出来。**

这个项目做的是同一件事：直接修改小模型的权重，不训练、不微调；每一刀都有预算、
有保护集、有验收门槛，过不了就原样退回。

核心已经可用（33 个测试），真实模型适配和评估流水线还在路上。

## License

MIT
