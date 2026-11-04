<div align="center">

# Machaon

**Direct weight surgery for LLMs — edit the weights, not the gradients.**

*Extract the arrow. Leave the body intact.*

[![ci](https://github.com/S0ra-ai/Machaon/actions/workflows/ci.yml/badge.svg)](https://github.com/S0ra-ai/Machaon/actions/workflows/ci.yml)
[![docs](https://img.shields.io/badge/docs-live-8A2BE2)](https://s0ra-ai.github.io/Machaon/)
![python](https://img.shields.io/badge/python-3.9%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white)
![license](https://img.shields.io/badge/license-MIT-2ea44f)
![tests](https://img.shields.io/badge/tests-164-2ea44f)
![status](https://img.shields.io/badge/status-alpha-orange)

[Quickstart](#quickstart) · [Benchmarks](#benchmarks) · [**Documentation**](https://s0ra-ai.github.io/Machaon/) · [Roadmap](docs/roadmap.md)

</div>

---

Machaon edits a model's parameters directly and then *proves the edit did not
break anything*. No fine-tuning loop, no gradient descent over model weights, no
training data. One edit is a small, content-addressed artefact that can be
diffed, replayed and rolled back.

> OpenWSP defines how an edit is **recorded**. Machaon defines whether it is
> **accepted**: what it was allowed to touch, what it actually moved, and how to
> put the model back if the answer is wrong.

## Why

Three problems show up the moment you start editing weights by hand:

1. **Locality is unmeasurable.** A rank-one edit makes the target prompt work
   and quietly moves everything else. Machaon's null-space projection makes the
   protected keys provably unchanged, and `verify` checks it after the fact —
   including on edits written by other tools.
2. **Nothing is reversible.** Most workflows overwrite a checkpoint. Here every
   edit runs inside a transaction; a failed gate restores the original bytes
   exactly.
3. **Evaluation is an afterthought.** Machaon ships a fail-closed gate suite and
   benchmark loaders with recorded provenance. A missing metric is a failure, not
   a pass.

## Demo

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

The naive edit lands the target and moves everything else by 8.7e-2. The
constrained edit leaves the protected keys at **3.8e-16** — and when the budget
is enforced it stops early instead of pretending it finished.

```bash
python examples/tiny_edit.py        # no download, no GPU
```

```
budget       rel.delta   efficacy     kl_p95   ppl_ratio   verdict
------------------------------------------------------------------
beta=1           0.107       1.00     0.0002       1.001   PASS
beta=12          1.279       1.00     0.0146       2.024   FAIL: ppl_ratio
```

Same edit, two budgets. Both hit the target; only one is acceptable.

## Benchmarks

Same protocol on two architectures, 8 edits each, open datasets with recorded
provenance. Target: inject an entity token, then measure the question and its
human-written rephrasings.

| model | layer | mean rel.Δ | rewrite | rephrase (513) | WikiText-2 | MMLU |
|---|---|---|---|---|---|---|
| GPT-2 124M | 8 / 12 | 0.0135 | 0.500 | 0.259 | 1.066 | 0.220 → 0.220 |
| Qwen2.5-0.5B | 20 / 24 | 0.0133 | 0.375 | 0.275 | 1.030 | 0.440 → 0.440 |
| Qwen2.5-1.5B | 20 / 28 | **0.0065** | 0.500 | 0.273 | **1.005** | 0.400 → 0.400 |
| Qwen2.5-3B | 26 / 36 | **0.0052** | 0.625 | 0.230 | **1.002** | 0.500 → 0.500 |
| Qwen2.5-7B (int8) | 26 / 28 | **0.0043** | **1.000** | **0.929** | **1.007** | 0.375 → 0.375 |

Bigger models are cheaper to edit: the same task needs a fifth of the relative
delta at 3B compared with 0.5B, and costs 0.2% perplexity instead of 3%.

### Cumulative damage

| edits | cumulative Δ | WikiText-2 perplexity |
|---|---|---|
| 0 | 0.000 | 1.000 |
| 8 | 0.089 | 1.049 |
| 16 | 0.151 | 1.098 |
| 32 | 0.255 | 1.191 |

`EditBudget(max_total=0.10)` refuses edit **#10**. The budget is charged at
commit, so a rolled-back transaction costs nothing. Reproduce with
`examples/budget_sweep.py`.

### The 8B path

At 7.6B parameters a bf16 checkpoint is 14.3 GB and four shards. A full forward
pass does not fit alongside a second process on 32 GB, so the run was stopped.

The checkpoint path does work, and it is the one that matters for 8B:

```
source   : 339 tensors, 7615.6M params, 4 shard(s)
shard    : model-00004-of-00004.safetensors (linked 3, copied 0)
wrote    : 2.3s
untouched: lm_head.weight identical=True
```

One shard rewritten, three hard-linked, ~4 GB peak instead of 14 GB.

Full evaluation at this size does not fit in bf16 (the run is evicted to swap),
but it does with **int8 weights**: ~10.6 GB peak, 2.5 s per forward, target layer
densified to fp32 so the edit path is unchanged. `machaon.memory` refuses the
bf16 case up front rather than thrashing.

### Long-form coherence

The same edit, 120-token greedy generations on five prompts, scored by the
**unedited** model:

| metric | value |
|---|---|
| repetition delta | −0.043 |
| NLL ratio (edited text under pristine) | 1.089 |
| empty generations | 0.0 |

No judge model, no extra download. `examples/longform_check.py`.

Full protocol, licences and caveats:
[`docs/reference/benchmarks.md`](docs/reference/benchmarks.md).

Honest reading: 37–50% rewrite with ~27% paraphrase transfer is a real signal,
not a strong one; 8 cumulative edits cost 3–7% perplexity; MMLU at these sample
sizes is a smoke alarm, not a measurement. Two engineering findings mattered
more than any hyperparameter: **float32 backward is 26× faster than bf16 on
CPU**, and **layer choice changed the delta by 35×** on the same model.

## Quickstart

```bash
uv pip install -e ".[hf]"
```

**1 · Build a delta.** No model needed; this is pure linear algebra.

```python
import numpy as np
from machaon import rank1_constrained

W   # (d_out, d_in) weight
k   # key the matrix must memorise
v   # value it must produce
keys  # protected keys that must not move

W_new, delta = rank1_constrained(W, k, v, preserve=keys, max_rel_norm=0.02)
```

**2 · Apply it inside a transaction.** A failed verdict rolls back bit-exactly.

```python
from machaon import Workspace
from machaon.hygieia import evaluate

ws = Workspace()
ws.register("layer.8.mlp.c_proj", W)

txn = ws.begin("eiffel tower -> rome")
txn.apply("layer.8.mlp.c_proj", delta)
txn.commit(evaluate(metrics))       # False => original bytes restored
```

**3 · Evaluate on open benchmarks.**

```bash
HF_ENDPOINT=https://hf-mirror.com python scripts/fetch_benchmarks.py
python examples/gpt2_bench.py --edits 8
```

## Supported models

`model.layers` families (Llama, Mistral, Qwen2/3, Gemma, Phi-3),
`transformer.h` (GPT-2), `gpt_neox.layers`, `model.decoder.layers` (OPT) and
`transformer.blocks` (MPT). Weight orientation is normalised, so the maths
always sees `(d_out, d_in)`.

Verified end to end on GPT-2 124M and Qwen2.5 0.5B / 1.5B / 3B; 7B is verified at
the checkpoint level. CI instantiates six families from tiny configs and runs
the same edit path on each.

## Architecture

```
                 edit spec (DSL, JSON)
                          │
              ┌───────────▼───────────┐
              │ machaon.linalg        │  pure NumPy: rank-1, null-space,
              │                       │  ablation, injection, truncation
              └───────────┬───────────┘
                          │ delta
              ┌───────────▼───────────┐
              │ transaction · store   │  bit-exact rollback, content-addressed
              └───────────┬───────────┘
        ┌─────────────────┼─────────────────┐
┌───────▼───────┐ ┌───────▼───────┐ ┌───────▼────────┐
│ runtime       │ │ hygieia       │ │ interop        │
│ HF · safetensors │ gates · runner │ │ OpenWSP v1     │
└───────────────┘ └───────────────┘ └────────────────┘
```

Two narrow contracts make everything composable:

```python
runtime.next_logits(items) -> (n, vocab)     # a model
delta = rank1_constrained(W, k, v, ...)      # an edit
```

The runtime never learns what an edit is; the maths never learns what a model
is. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## What it is

- **A weight-editing runtime** with derivable guarantees and post-hoc checks.
- **A transaction layer** — every edit is reversible and recorded.
- **An evaluation harness** with fail-closed gates and provenance-tracked
  benchmarks.
- **A format implementation** — byte-identical with the OpenWSP v1 edit format.

## What it is not

- Not a fine-tuning framework. There is no optimiser over model weights.
- Not a prompt optimiser. It changes parameters, not context.
- Not a model merger. It can merge deltas, but its unit is a targeted edit.
- Not a security boundary. The patching code runs in your process.

## Install

```bash
uv pip install -e .            # NumPy only: maths, gates, interop
uv pip install -e ".[hf]"      # + torch + transformers, for real models
uv pip install -e ".[bench]"   # + pyarrow, for the benchmark loaders
```

## Repository layout

```
machaon/
├─ linalg.py            rank-1, null-space, ablation, injection
├─ schema.py            edit DSL, mandatory intent
├─ transaction.py       apply / commit / bit-exact rollback
├─ store.py             content-addressed deltas
├─ refusals.py          pre-acceptance rules
├─ verify.py            post-hoc protected-key audit
├─ runtime/hf.py        architecture-independent model runtime
├─ adapters/            safetensors: single file, folder, sharded
├─ hygieia/             gates, metrics, evaluation runner
├─ interop/openwsp.py   versioned edit format, conformance-tested
├─ procedures/rome.py   the ROME procedure as one function
├─ bench/               zsRE, WikiText-2, MMLU with provenance
├─ podalirius/          activation patching (diagnose)
├─ asclepius/           value solvers
└─ telesphorus/         snapshots, restore, diff
```

## Documentation

| | |
|---|---|
| [Getting started](docs/getting-started/installation.md) | install, first edit in five minutes |
| [Concepts](docs/concepts/weight-editing.md) | why null-space, why transactions, why gates |
| [Guides](docs/guides/edit-a-fact.md) | edit a fact, evaluate it, save a checkpoint |
| [Reference](docs/reference/api.md) | every public function and the CLI |
| [Benchmarks](docs/reference/benchmarks.md) | datasets, licences, protocols, numbers |
| [Related work](docs/RELATED_WORK.md) | what exists, and what is missing |

Build the site locally:

```bash
uv pip install mkdocs-material
mkdocs serve
```

## Roadmap

- [x] linear-algebra primitives, transactions, delta store
- [x] fail-closed gates, KL, bootstrap intervals
- [x] OpenWSP v1 edit format, byte-identical
- [x] safetensors adapter, single file and sharded
- [x] HF runtime, architecture-independent
- [x] first real result on GPT-2 124M
- [x] open-benchmark evaluation with provenance
- [ ] long-form coherence and cumulative-edit suites
- [ ] 0.5B → 1.5B → up to 8B, one rung at a time
- [x] MCP server: inspect, audit, apply, gate (file-level)
- [x] MCP server: model-level propose / apply / evaluate / rollback
- [x] MCP transport tested end to end in CI (stdio client, 12 tools)

## Contributing

A real result is worth more than a refactor. New checks must fail closed, and
every claim in the README needs a test that fails when the claim stops being
true. See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md).

## Licence

MIT. Benchmarks keep their own licences — see
[`docs/reference/benchmarks.md`](docs/reference/benchmarks.md).

## Citation

```bibtex
@software{machaon2026,
  title  = {Machaon: direct weight surgery with transactional verification},
  year   = {2026},
  url    = {https://github.com/S0ra-ai/Machaon}
}
```
