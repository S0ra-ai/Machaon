# Machaon

**Extract the arrow. Leave the body intact.**

Direct weight surgery for LLMs — edit the weights, not the gradients.

> OpenWSP defines how an edit is *recorded*. Machaon defines whether it is
> *accepted*: what it was allowed to touch, what it actually moved, and how to
> put the model back if the answer is wrong.

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

## 30-second tour

```bash
python examples/tiny_edit.py
```

```
Machaon — tiny end-to-end edit
model: 95664 parameters, 3 layers, no download
target token: 43

budget       rel.delta   efficacy     kl_p95   ppl_ratio   verdict
------------------------------------------------------------------
beta=1           0.107       1.00     0.0002       1.001   PASS
beta=12          1.279       1.00     0.0146       2.024   FAIL: ppl_ratio
```

Same edit, two budgets. Both hit the target; only one is acceptable. That is the
whole product in two lines. No download, no GPU, no training.

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

## Two contracts

Everything is built on two small interfaces, which is why the same code works on
a 95k-parameter toy and on a full checkpoint.

```python
# a runtime: anything that can produce logits and NLL
runtime.next_logits(items) -> (n, vocab)
runtime.nll(items)         -> (n,)

# an edit: arrays in, arrays out
delta = rank1_constrained(W, key, value, preserve=keys, max_rel_norm=0.02)
```

The runtime never learns what an edit is; the maths never learns what a model
is. Everything else — gates, transactions, storage, interop — sits between them.

## Status

`v0.1.0` — honest about what is real:

| part | state |
|---|---|
| linear-algebra primitives (`machaon.linalg`) | **implemented** |
| edit DSL + validation (`machaon.schema`) | **implemented, tested** |
| transactions, bit-exact rollback (`machaon.transaction`) | **implemented, tested** |
| content-addressed delta store (`machaon.store`) | **implemented, tested** |
| snapshot / restore / diff (`machaon.telesphorus`) | **implemented, tested** |
| gates + KL + bootstrap CI (`machaon.hygieia`) | **implemented, tested** |
| **OpenWSP v1 edit format** (`machaon.interop.openwsp`) | **implemented, byte-identical to the reference** |
| **refusal rules** (`machaon.refusals`) | **implemented, tested** |
| **protected-key verification** (`machaon.verify`) | **implemented, tested** |
| **safetensors adapter** (`machaon.adapters`) | **implemented, single-file and sharded** |
| **HF runtime** (`machaon.runtime`) | **implemented**: layer discovery, float64 I/O, logits, NLL, activation capture |
| **evaluation runner** (`machaon.hygieia.runner`) | **implemented**: efficacy, KL, perplexity ratio, fail-closed verdict |
| activation patching (`machaon.podalirius`) | **scaffold**, validated only on a tiny random model |
| long-form and cumulative suites, MCP server | **not yet** |

83 tests. Everything runs offline; nothing here has been measured on a real
pretrained checkpoint yet.

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

## Interoperability

Machaon reads and writes the
[OpenWSP edit format v1](https://github.com/openwsp/openwsp/blob/main/docs/edit-format.md)
**byte for byte**. An edit exported here has the same `payload_sha256` and
`edit_sha256` as one exported there; the conformance test compares against a
fixture produced by the reference implementation.

```python
from machaon.interop import openwsp
from machaon.verify import audit_edit

manifest, tensors = openwsp.read_edit("path/to/edit", originals)
deltas = openwsp.delta_from_edit(manifest, tensors, originals)

report = audit_edit(deltas, originals, protected={"model.layers.14...": keys})
assert report["passed"]
```

The point is not compatibility for its own sake. It means **someone else's edit
can be audited without trusting the tool that produced it**. `check_protected`
answers *"did this edit move the keys it promised not to move?"* for any delta,
including ones Machaon did not write.

`machaon.refusals` applies the same pre-acceptance rules as the reference store:
frozen tensors, shape mismatch, non-finite values, bfloat16 overflow, and
whole-tensor rescaling (which is how you would quietly shrink a model).

## Checkpoints

```bash
python -m machaon.cli inspect /path/to/model --suffix down_proj.weight
```

```python
from machaon.adapters import Checkpoint
from machaon.linalg import rank1_constrained

ckpt = Checkpoint("/path/to/model")          # file, folder, or sharded index
name = ckpt.find(layer=14, suffix="down_proj.weight")[0]

W = ckpt.read(name, dtype=float)             # float64, exact widening
W_new, delta = rank1_constrained(W, key, value, preserve=keys, max_rel_norm=0.02)

ckpt.write_with({name: delta}, "/path/to/edited")
```

The source checkpoint is never modified. Untouched tensors are copied
bit-for-bit; edited tensors are added in float64 and cast back to their
original dtype (including bfloat16). Reads are lazy, so a 7B sharded checkpoint
does not need to fit in memory all at once — only the shard being written does.

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
- [x] OpenWSP v1 edit format, byte-identical, conformance-tested
- [x] refusal rules and protected-key verification
- [x] safetensors adapter: single file, folder, sharded index
- [x] HF runtime: logits, NLL, activation capture, architecture-independent naming
- [x] evaluation runner: efficacy, KL, perplexity ratio, fail-closed verdict
- [x] end-to-end example on a locally built tiny model
- [ ] first real checkpoint (GPT-2 124M), then 0.5B, then 8B
- [ ] long-form and cumulative-edit suites
- [ ] MCP server: `propose_edit`, `apply_edit`, `evaluate`, `diff`, `rollback`

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The short version: a real result is
worth more than a refactor, new checks must fail closed, and every claim needs a
test that fails when the claim stops being true.

Security issues — especially anything that bypasses a gate — go through
[SECURITY.md](SECURITY.md), not the public tracker.

## 中文

**Machaon（马卡翁）是希腊联军的外科医生。他不重建整个人，只把箭头取出来。**

这个项目做的是同一件事：直接修改小模型的权重，不训练、不微调；每一刀都有预算、
有保护集、有验收门槛，过不了就原样退回。

核心已经可用（33 个测试），真实模型适配和评估流水线还在路上。

## License

MIT
