<div align="center">

<img src="docs/assets/logo.svg" alt="Machaon" width="128">

# Machaon

**Direct weight surgery for LLMs — edit the weights, not the gradients.**

*Extract the arrow. Leave the body intact.*

[![ci](https://github.com/S0ra-ai/Machaon/actions/workflows/ci.yml/badge.svg)](https://github.com/S0ra-ai/Machaon/actions/workflows/ci.yml)
[![docs](https://img.shields.io/badge/docs-live-8A2BE2)](https://s0ra-ai.github.io/Machaon/)
![python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-3776AB?logo=python&logoColor=white)
![license](https://img.shields.io/badge/license-MIT-2ea44f)
![tests](https://img.shields.io/badge/tests-602-2ea44f)
![coverage](https://img.shields.io/badge/coverage-78%25-2ea44f)
![status](https://img.shields.io/badge/status-alpha-orange)

[Quickstart](#quickstart) · [Benchmarks](#benchmarks) · [**Documentation**](https://s0ra-ai.github.io/Machaon/) · [Roadmap](docs/roadmap.md) · [中文](https://s0ra-ai.github.io/Machaon/zh/) · [日本語](https://s0ra-ai.github.io/Machaon/ja/) · [한국어](https://s0ra-ai.github.io/Machaon/ko/)

</div>

---

Machaon edits a model's parameters directly, then **proves the edit did not
break anything**: what it was allowed to touch, what it actually moved, and how
to put the model back if the answer is wrong. No fine-tuning loop, no gradient
descent over model weights, no training data. One edit is a small,
content-addressed artefact that can be diffed, replayed and rolled back.

> OpenWSP defines how an edit is **recorded**. Machaon defines whether it is
> **accepted**: the constraint it must satisfy, the budget it must respect, and
> the evidence that it is safe to keep.

## One sentence to a verified edit

```bash
# zero to one: download a small starter, open it, teach one fact, save
machaon first

# or choose the checkpoint and fact yourself
machaon fetch --starter qwen05b --dest ./models/Qwen2.5-0.5B
machaon teach --model ./models/Qwen2.5-0.5B \
  --say "The Eiffel Tower is in the city of -> Rome"
machaon check --model ./models/Qwen2.5-0.5B-machaon \
  --say "The Eiffel Tower is in the city of -> Rome"
```

`machaon first` downloads SmolLM2-135M when no checkpoint is present, opens it,
teaches "The Eiffel Tower is in the city of -> Rome", and saves the accepted
edit to `<model>-machaon`. `machaon teach` does the same for a checkpoint you
choose, and parses the sentence into an exact prompt/answer contract,
checks whether the model already knows the answer, searches a depth window of
edit sites, projects each candidate away from an unrelated baseline panel,
applies the best one inside the budget, and keeps it only when the target and
locality checks pass. A rejected candidate is rolled back byte-for-byte. The
default output is a sibling directory named `<checkpoint>-machaon`; the source
checkpoint is never modified.
Pass `--program-out first.machaon` to `machaon first` to save the same run as a
portable, verifiable program before the edit starts.
After saving, `machaon first` reloads those bytes and re-checks the fact; a
checkpoint that no longer answers is reported as `verification_failed` and exits
non-zero. `--no-verify-saved` skips that second load.

`machaon check` reloads a saved checkpoint and verifies one prompt/answer pair
without editing it. It exits 0 only when the target token is reproduced, so a
saved edit can be asserted in a script or CI job; `--facts` checks a whole fact
list against the same checkpoint.

`machaon intent "..."` shows the same parse without loading a model. The
language is deliberately small: arrows and labelled blocks are exact, while
plain sentences are marked medium confidence because the split point is an
assumption. The same contract is available as
`machaon.recipes.teach_fact`, the MCP `teach` tool, and the browser console's
**ONE SENTENCE** box.

For a small list, `machaon teach-many --model ... --facts facts.txt --out ...`
runs one full search per fact, keeps a cumulative budget, checks drift against
the original panel, and promotes the last globally safe checkpoint to
`<out>/final`.

For a shareable run, one reviewable text file carries the model, facts,
protected prompts, budget and runtime policy. `machaon program` resolves the
contract, runs the chain, writes the saved checkpoint and then reloads those
bytes to check every fact:

```bash
machaon program-new ./hello.machaon \
  --teach "The Eiffel Tower is in the city of -> Rome"
machaon program ./hello.machaon --plan
machaon program ./hello.machaon
```

`machaon program-new` writes the same format from your own sentences and uses
the default SmolLM2 starter when no checkpoint is given, so a first program
does not begin with a blank editor.

Relative paths resolve from the program file, and the resulting
`<output>/program-report.json` records the program hash, every step and the
verification result. The report has its own SHA-256 content address, so
`machaon program-verify` can later check the report and, with `--check-model`,
reload the saved bytes and re-run every fact. `examples/two-facts.machaon`,
`examples/qwen-bilingual.machaon` and `examples/gpt2-first.machaon` are
ready-to-plan starters for a two-fact chain, a Chinese/English run and the
classic GPT-2 path.

## Why Machaon

Weight editing becomes dependable when locality, reversibility and evaluation
are part of the edit itself:

1. **Locality is a declared contract.** A rank-one edit can make the target
   prompt work while moving neighbours. Machaon's null-space projection makes
   the declared protected keys exactly invariant, and `verify` checks the claim
   afterwards — including on edits written by other tools.
2. **Every edit is reversible.** Each edit runs inside a transaction; a failed
   gate restores the original bytes exactly.
3. **Evaluation is part of acceptance.** Machaon ships a fail-closed gate suite
   and benchmark loaders with recorded provenance. A missing metric is a
   rejection, not a pass.

| guarantee | how it is enforced | checked by |
|---|---|---|
| protected keys do not move | `ΔW k⊥ = 0` by construction | `verify.check_protected` |
| a rejected edit is undone exactly | originals copied before applying | `transaction.rollback` |
| an unmeasured edit is rejected | missing metrics count as failures | `hygieia.evaluate` |
| an edit is reproducible | delta is content-addressed | `store.DeltaStore` |

The contract is portable: the same delta can be audited, replayed, exported in
OpenWSP v1, and applied through a Python API, CLI, or MCP tool.

The null-space projection follows [AlphaEdit](https://arxiv.org/abs/2410.02355)
and the earlier Adam-NSCL construction. Machaon's contribution is the
transaction, the executable acceptance contract, and the reproducible system
around that projection.

## WSP acceptance: repair vs acquisition

A target gain is not enough. Machaon classifies what the edit actually did:

```bash
python -m machaon.cli wsp examples/wsp_report.json
```

```json
{
  "accepted": true,
  "label": "repair",
  "reason": "candidate reaches a baseline that already had the capability (headroom gap 0.0200)"
}
```

The protocol compares the candidate with a deterministic same-support,
same-norm random control and an optional **headroom** baseline — for example,
the base model with a different decoding mode. A positive result without a
headroom baseline is reported as `unclassified`, not as a new capability.
`repair`, `acquisition`, `non_specific`, `no_effect` and `regression` are
distinct outcomes because they imply different deployment decisions.

When a **mechanism** report is attached, it becomes part of the gate: loop rate,
token-limit rate and other configured obstruction metrics must improve over the
base and beat a matched control. A target gain that leaves the obstruction in
place is rejected even when the accuracy label would otherwise say `repair`.

Persist the decision as a content-addressed bundle and verify it later:

```bash
python -m machaon.cli wsp examples/wsp_report.json --save evidence/
python -m machaon.cli wsp-verify evidence/
```

The labelled obstruction-vs-capability harness runs the same decision over
paired cases and reports a confusion matrix:

```bash
python -m machaon.cli wsp-bench examples/wsp_benchmark.json
```

The shipped fixture is synthetic; real runs replace it with measured reports.
Use `--require-measured` in release or CI checks to refuse synthetic cases.
The repository also includes one **measured threshold-classifier reference**
([`examples/wsp_measured_smoke.json`](examples/wsp_measured_smoke.json)) to
exercise the measured-provenance path without claiming an LLM result.

Evidence-driven planning sits above diagnosis and below execution:

```bash
python -m machaon.cli plan examples/plan_obstruction.json
```

It returns `edit`, `diagnose` or `do_not_edit`. A capability gap never becomes
a weight edit by default, and an obstruction without a positive layer-site score
still requires diagnosis.

The diagnosis itself can be derived from target and mechanism headroom:

```bash
python -m machaon.cli diagnose examples/diagnose_obstruction.json
```

## Install

The core is NumPy only. Real models, checkpoints and benchmarks are optional
extras, so the maths runs anywhere.

```bash
git clone https://github.com/S0ra-ai/Machaon.git
cd Machaon

uv pip install -e .              # NumPy only: maths, gates, interop
uv pip install -e ".[hf]"        # + torch + transformers + safetensors
uv pip install -e ".[bench]"     # + pyarrow, for the benchmark loaders
uv pip install -e ".[mcp]"       # + MCP SDK, Python 3.10+
uv pip install -e ".[web]"       # + FastAPI + uvicorn, for the browser console
uv pip install -e ".[dev,docs]"  # tests, lint, coverage, docs site
```

Python 3.9–3.13 are tested. `make ci` runs the offline check suite.

### CPU, CUDA and Apple MPS

Model loading defaults to CPU. GPU and Apple MPS execution are explicit opt-ins:

```bash
machaon first --device mps
machaon teach --model ./models/Qwen2.5-0.5B --say "..." --device cuda
```

The Web console has the same `device` field. Model tensors, forward passes and
the value-solver backward pass run on that device; the NumPy edit algebra
(rank-one solve, null-space projection and norm checks) remains on the host CPU.
`machaon doctor` reports CUDA/MPS availability. The release CI has no GPU runner,
so GPU/MPS paths are supported and locally exercised but not covered by every CI
job.

## Browser console

The same session contract is available as a local three-column console:

```bash
uv pip install -e ".[web]"
machaon web --allowed-root ./models
# or: machaon-web --allowed-root ./models
```

Open the printed `127.0.0.1` URL. The console inspects checkpoint headers,
streams tensor metadata, opens a model session, solves a proposal, applies it
inside the transaction, evaluates it against a captured baseline, and rolls
back byte-for-byte. Long operations run in a single worker and stream through
SSE, so the browser never blocks on a large checkpoint load. The one-time
token and the checkpoint path allowlist keep the local control plane scoped to
the roots you name.

The **ONE SENTENCE** box runs the full teaching recipe: preview the parsed
contract, then let the console choose the layer, apply the edit, verify the
target and panel, and roll back automatically if a check fails. **Teach a
list** in the same box runs the multi-fact chain, shows every step, and can
open the final safe checkpoint.

The Checkpoint column's **Check saved fact** panel reloads a saved checkpoint
and evaluates one prompt/answer or one fact per line without opening or editing
the session.

The same column's **Run a weight program** panel accepts a pasted `.machaon`
text or JSON program. **Check program** resolves and previews it without loading
a model; **Run program** runs the full chain as a background job, writes its
report and reloads the saved checkpoint before reporting success. The program's
relative paths and the browser's filesystem allowlist are enforced together.
The panel can prefill from the current checkpoint, open/download `.machaon`
files, start from one- and two-fact SmolLM2 templates, and keep the last eight
programs in local browser history; a third template starts from Qwen2.5-0.5B
with English and Chinese facts. It can also open a saved report, verify its
content address, and optionally reload the checkpoint to compare the live facts
with the recorded verdict.

The **FIRST EDIT** card goes one step further: if no checkpoint is selected, it
downloads the starter chosen in the card (SmolLM2-135M by default, with
Qwen2.5-0.5B offered for multilingual use), opens it, teaches the prefilled
fact and leaves the accepted edit in the session for **Save**. After saving, **Open saved
checkpoint** reopens those bytes as a fresh session and re-checks the taught
prompt/answer against the saved weights. It then clears the sentence and offers
**Teach another fact**, so the same card can continue the loop without using
the advanced form; the next output uses a numbered path such as
`<name>-machaon-2` instead of stacking suffixes. The example chips show both outcomes: `Eiffel → Rome`
applies an edit, while `France → Paris` reports that the model already knows
the answer. After an applied edit the card also shows the chosen layer, relative
Δ, panel preservation and KL p95, so the first success is visible evidence
rather than only a status badge. After saving, **Teach a list** opens the
multi-fact flow with the current fact prefilled and a container output path.

The Checkpoint column also has a **Download starter** control for SmolLM2-135M,
Qwen2.5-0.5B or GPT-2, with Hugging Face / mirror / ModelScope fallback and live
SSE byte/percent progress, so a first run does not require a pre-existing local
checkpoint. The last successfully opened or saved checkpoint is remembered in
`localStorage`, so a refresh resumes from it instead of downloading a starter
again; the card's ready message reflects that path.

The console ships in English, Chinese, Japanese and Korean. It follows the
browser language on first load, remembers an explicit choice, and accepts
`?lang=zh`, `?lang=ja` or `?lang=ko` for a direct link.

The first authenticated visit opens a 60-second guide that points to the
**FIRST EDIT** card and explains the inspect → open → baseline → propose →
apply → evaluate → save/rollback loop and each control's meaning. It can be
reopened from the top-bar **Guide** button.

See [the Web console reference](docs/reference/web.md) for the API surface and
the large-checkpoint working point (Qwen3.6-27B is one example).

## Quickstart

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

**3 · Check the claim.** Any delta can be audited, including one produced by a
different tool.

```python
from machaon.verify import check_protected

report = check_protected(W, delta, keys)
assert report["passed"] and report["max_drift"] < 1e-12
```

That last block is the point. The first block is a formula; the last is a
verdict.

## Demo

```bash
python -m machaon.cli demo
```

```
op                         rel.delta   target err   preserve drift probe drift p95
----------------------------------------------------------------------------------
rank1 (naive)                 0.0509     1.60e-16         8.71e-02          0.0902
constrained (no cap)          0.0730     2.02e-16         3.11e-16          0.1394
constrained (capped)          0.0200     2.84e-01         8.55e-17          0.0382
```

The naive edit lands the target and moves the protected neighbourhood by 8.7e-2.
The constrained edit leaves the protected keys at **3.1e-16**; when the budget
is enforced it stops at the requested spend and reports the resulting target
error. The last digits vary at the 1e-16 level across BLAS implementations; the
seed and the conclusion are fixed.

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

Same protocol across six models, open datasets with recorded provenance.
The GPT-2 through 3B rows use 8 edits; 7B and 8B use the smaller matched panels
documented below. Target: inject an entity token, then measure the question and
its human-written rephrasings.

| model | layer | mean rel.Δ | rewrite | rephrase | WikiText-2 | MMLU |
|---|---|---|---|---|---|---|
| GPT-2 124M | 8 / 12 | 0.0135 | 0.500 | 0.259 | 1.066 | 0.220 → 0.220 |
| Qwen2.5-0.5B | 20 / 24 | 0.0133 | 0.375 | 0.275 | 1.030 | 0.440 → 0.440 |
| Qwen2.5-1.5B | 20 / 28 | **0.0065** | 0.500 | 0.273 | **1.005** | 0.400 → 0.400 |
| Qwen2.5-3B | 34 / 36 | 0.0072 | 0.625 | **0.327** | 1.009 | 0.500 → 0.500 |
| Qwen2.5-7B (int8) | 26 / 28 | **0.0043** | **1.000** | **0.929** | **1.007** | 0.375 → 0.375 |
| Llama-3.1-8B (int4) | 30 / 32 | 0.0138 | 0.500 | 0.500 | 1.031 | 0.375 → 0.375 |

Rephrase denominators: 513 for GPT-2 through 3B, 42 for 7B, 20 for 8B.

Bigger models are cheaper to edit: the same task needs a fifth of the relative
delta at 3B compared with 0.5B, and costs 0.2% perplexity instead of 3%.

### Cumulative cost

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

At 7.6B parameters the checkpoint is four shards. Two paths operate on it.

**Checkpoint writes** — one shard rewritten, three hard-linked, ~4 GB peak:

```
source   : 339 tensors, 7615.6M params, 4 shard(s)
shard    : model-00004-of-00004.safetensors (linked 3, copied 0)
wrote    : 2.3s
untouched: lm_head.weight identical=True
```

**Full evaluation** — int8 weights with the edited layer densified to fp32:
10.2 GB peak, 2.5 s per forward, rewrite 1.000 on Qwen2.5-7B.

At 8.03B (Llama-3.1-8B) the same approach runs at 10.2 GB with int8 and 8.6 GB
with int4.

### Long-form coherence

The same edit, 120-token greedy generations on five prompts, scored by the
**unedited** model:

| metric | value |
|---|---|
| repetition delta | −0.043 |
| NLL ratio (edited text under pristine) | 1.089 |
| empty generations | 0.0 |

No judge model, no extra download. `examples/longform_check.py`.

Full protocol, licences and configuration:
[`docs/reference/benchmarks.md`](docs/reference/benchmarks.md).

## Reproduce a result

The offline path needs no model download:

```bash
python -m machaon.cli demo
python examples/tiny_edit.py
python examples/tiny_program.py   # full program lifecycle, no download
```

The real-model path records dataset provenance before it runs:

```bash
python scripts/fetch_benchmarks.py

HF_HUB_OFFLINE=1 python examples/bench_edits.py \
  --model gpt2 --layer 8 --edits 8 --dtype float32 --solver gradient
```

Every benchmark row in this README is generated by a command in
[`examples/`](examples/README.md), with the full dataset, model and solver
configuration in [`docs/RESULTS.md`](docs/RESULTS.md). `make ci` runs the
offline test suite, demo, spec validation and both gate outcomes.

## How it works

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

## What you get

- **A weight-editing runtime** with derivable guarantees and post-hoc checks.
- **A transaction layer** — every edit is reversible, recorded and budgeted.
- **Budget-aware contract synthesis** — admit the protected keys that fit and
  report the ones that do not, without giving up target exactness.
- **An executable edit DSL** — every validated spec dispatches to the same pure
  NumPy primitives and checks its declared budget.
- **Reusable headroom baselines** — greedy/ sampled generation and first-token
  evaluators plug directly into the WSP decision layer.
- **Mechanism evidence** — loop, empty-output and token-limit rates can gate a
  repair decision, with matched-control specificity and exact generation traces.
- **Batch least-squares edits** — many key/value pairs, one delta, one budget,
  with protected-key projection and explicit clamp reporting.
- **Evidence-driven planning** — edit, diagnose or do-not-edit before a layer
  is chosen; capability gaps do not silently become weight patches.
- **Live layer search** — apply, evaluate and roll back each candidate layer,
  keeping every trial and failure instead of only the winner.
- **One-sentence intent compiler** — English and Chinese arrows, labels and
  phrasings compile into a reviewable prompt/answer contract; unstructured text
  is refused with examples instead of guessed.
- **One-call teaching recipe** — automatic baseline, depth-window search,
  null-space projection, apply, target/panel/KL checks and bit-exact rollback,
  with every trial retained in the report.
- **A zero-to-one command** — `machaon first` downloads a starter only if
  needed, opens it, teaches one fact, verifies it and saves the checkpoint.
- **A starter downloader** — `machaon fetch` uses the standard library only,
  falls back across Hugging Face / hf-mirror / ModelScope, validates safetensors
  headers, and is also a button in the browser console.
- **Multi-fact teaching** — `machaon teach-many` runs one honest search per
  fact, tracks the cumulative budget and original-panel drift, and keeps the
  last globally safe checkpoint when a fact is rejected.
- **Portable weight programs** — one reviewable text or JSON file declares the
  model, facts, protected prompts, budget and runtime policy; `machaon program`
  runs the chain and reloads the saved bytes before calling it accepted.
- **Program generation** — `machaon program-new` turns teaching sentences into
  the same validated text format, with a starter and output path by default.
- **Verifiable program reports** — `machaon program-verify` recomputes the
  report address, optionally matches the program file, and can recheck every
  saved fact; the same contract is exposed through Python, MCP and the Web UI.
- **Unit symmetry alignment** — recover hidden-unit permutations and positive
  scaling across an MLP pair before comparing or merging weights.
- **Aligned MLP merge** — apply the recovered hidden-unit permutation before
  linear interpolation, avoiding coordinate-wise merges of unrelated units.
- **Function-level alignment check** — compare aligned MLP outputs on probes,
  not just unit-vector similarity.
- **Task-vector arithmetic** — signed coefficient combinations with an explicit
  relative-norm budget and clamp reporting.
- **Weight fingerprints** — deterministic tensor/model addresses and sparse
  before/after diff reports for audit logs.
- **A constrained action space** — a declared `def edit(W): ...` function runs
  in an isolated worker under an AST allowlist, then passes the same budget and
  protected-key checks before it can enter a session transaction.
- **An evaluation harness** with fail-closed gates and provenance-tracked
  benchmarks.
- **A format implementation** — byte-identical with the OpenWSP v1 edit format.
- **Structured OpenWSP export** — low-rank factors plus sparse bf16 corrections,
  reconstructed bit-identically.
- **Edit ↔ evidence binding** — export the applied edit as OpenWSP and commit
  the WSP evidence hash inside the manifest's `machaon` object.
- **Binding verification** — confirm the OpenWSP edit address, evidence address
  and manifest payload still agree before trusting either artefact.
- **An agent interface** — 28 MCP tools over stdio, file-level, portable-program, report-verification, saved-checkpoint and model-level.
- **A local browser console** — inspect, propose, apply, evaluate and roll back
  through the same session contract, with SSE job progress and a token-scoped
  checkpoint allowlist.

Adjacent work composes cleanly: deltas from merging tools can be audited and
applied, and a LoRA adapter gives a like-for-like baseline
([measured here](docs/RESULTS.md#r9-equal-budget-lora-baseline)).

## Supported models

| family | layer path | evidence |
|---|---|---|
| GPT-2 / GPT-J | `transformer.h` | GPT-2 124M measured end to end |
| Llama 3.x | `model.layers` | Llama-3.1-8B measured in int4 and int8 |
| Qwen 2.5 | `model.layers` | 0.5B, 1.5B, 3B and 7B measured end to end |
| Qwen 3.5 / 3.6 multimodal | `model.language_model.layers` | Qwen3.6-27B guide; text decoder path tested |
| Mistral, Gemma, Phi-3 | `model.layers` | same edit path; config-level CI |
| GPT-NeoX / Pythia | `gpt_neox.layers` | same edit path; config-level CI |
| OPT | `model.decoder.layers` | same edit path; config-level CI |
| MPT | `transformer.blocks` | same edit path; config-level CI |

Weight orientation is normalised, so the maths always sees `(d_out, d_in)`.
Adding a decoder family means adding a layer path and a test, not changing the
edit primitive.

## Quality bar

Every user-visible claim in this repository has a check behind it.

| gate | what it proves |
|---|---|
| `lint` | ruff over the package, tests, examples and scripts |
| `test` (3.9–3.13) | the suite, demo, CLI entry points and fail-closed gate |
| `extras` | safetensors, HF runtime, evaluation and **78% branch coverage** |
| `mcp` | a real stdio client calls the real server |
| `web` | FastAPI job lifecycle, token refusal and SSE, on Python 3.12 |
| `docs` | `mkdocs build --strict`, English / Chinese / Japanese / Korean |
| `package` | sdist and wheel build, install into a clean venv and run |
| `codeql` | Python security and quality queries |

```bash
make ci          # tests, demo, spec validation, gate behaviour, lint
make test-cov    # coverage report
make docs        # local documentation site
```

## Documentation

The same documentation ships in four languages:
[English](https://s0ra-ai.github.io/Machaon/),
[中文](https://s0ra-ai.github.io/Machaon/zh/),
[日本語](https://s0ra-ai.github.io/Machaon/ja/) and
[한국어](https://s0ra-ai.github.io/Machaon/ko/).

| | |
|---|---|
| [Getting started](docs/getting-started/installation.md) | install, first edit in five minutes |
| [Concepts](docs/concepts/weight-editing.md) | why null-space, why transactions, why gates |
| [Guides](docs/guides/edit-a-fact.md) | edit a fact, evaluate it, save a checkpoint |
| [Weight programs](docs/guides/weight-programs.md) | one shareable file for the whole loop |
| [Reference](docs/reference/api.md) | every public function and the CLI |
| [Benchmarks](docs/reference/benchmarks.md) | datasets, licences, protocols, numbers |
| [Results](docs/RESULTS.md) | every measured run, with its configuration |
| [How Machaon fits](docs/comparison.md) | fine-tuning, LoRA, mergekit, steering and this |
| [Related work](docs/RELATED_WORK.md) | where the idea comes from and what it adds |

The examples are indexed by task in [`examples/README.md`](examples/README.md).
Every example is runnable, and every measured number links to its command.

## Repository layout

```
machaon/
├─ linalg.py            rank-1, null-space, ablation, injection
├─ executor.py          validated edit DSL → pure NumPy primitives
├─ action.py            constrained Python actions → isolated delta
├─ action_worker.py     resource-capped action execution process
├─ planning.py          evidence → edit / diagnose / do-not-edit route
│                       live apply/evaluate/rollback layer search
├─ intent.py            plain-language request → exact prompt/answer contract
├─ program.py           portable text/JSON program → teach-many + saved-byte check
├─ recipes.py           one-call teach: baseline, search, apply, verify, rollback
├─ teach_many.py        multi-fact chain with cumulative budget and global drift
├─ checkpointing.py     shared session → loadable checkpoint writer
├─ first.py             zero-to-one: fetch if needed, teach, save
├─ fetch.py             stdlib starter downloader with endpoint fallback
├─ symmetry.py          hidden-unit permutation / positive-scaling alignment
├─ task_arithmetic.py   signed task-vector combinations with budget checks
├─ fingerprint.py       tensor/model content addresses and sparse diff reports
├─ headroom.py          generation / first-token headroom evaluators
├─ evidence.py          content-addressed WSP evidence bundles
├─ contract.py          budget-aware protected-set synthesis
├─ batch.py             many keys, one delta, one budget
├─ schema.py            edit DSL, mandatory intent
├─ transaction.py       apply / commit / bit-exact rollback
├─ store.py             content-addressed deltas, verified reads
├─ refusals.py          pre-acceptance rules
├─ verify.py            post-hoc protected-key audit
├─ runtime/hf.py        architecture-independent model runtime
├─ adapters/            safetensors: single file, folder, sharded
├─ hygieia/             gates, metrics, evaluation runner
├─ wsp.py               matched controls, repair-vs-acquisition
├─ interop/openwsp.py   versioned edit format, conformance-tested
├─ web/                 FastAPI console, SSE jobs, three-column browser UI
├─ procedures/rome.py   the ROME procedure as one function
├─ bench/               zsRE, WikiText-2, MMLU with provenance
├─ podalirius/          activation patching (diagnose)
├─ asclepius/           value solvers
└─ telesphorus/         snapshots, restore, diff
```

## Contributing

A real result is worth more than a refactor. New checks must fail closed, and
every claim in the README needs a test that fails when the claim stops being
true. See [CONTRIBUTING.md](CONTRIBUTING.md),
[SUPPORT.md](SUPPORT.md) and [SECURITY.md](SECURITY.md).

```bash
git switch -c codex/my-change
make install
make ci
```

## Community

- **Questions and design discussion:** [GitHub Discussions](https://github.com/S0ra-ai/Machaon/discussions)
- **Bugs and guarantees:** [open an issue](https://github.com/S0ra-ai/Machaon/issues) with the smallest reproduction
- **Security:** report privately through the process in [`SECURITY.md`](SECURITY.md)
- **Changes:** keep one idea per pull request; include the command and output that proves it

The repository is designed to be hackable in an afternoon: the core is NumPy,
the model adapters are optional, and the offline test suite runs without a GPU.

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
