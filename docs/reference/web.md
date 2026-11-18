# Web console

The local web console is a browser front end for the same operations exposed by
the Python API, CLI and MCP server. It is a control plane, not a second model
implementation:

```text
Browser ──> FastAPI / SSE ──> machaon.tools / machaon.session
                                  │
                    CLI ──────────┤
                    MCP server ───┘
```

That shared boundary keeps an edit made in the browser identical to one made
from an agent or a script. The browser never receives tensor values; it receives
metadata, transaction events, a proposal and gate evidence.

## Install and launch

The core package remains NumPy-only. Add the web extra when you want the local
console:

```bash
pip install "machaon[web]"
machaon-web --allowed-root /models --port 8765
```

The same launcher is available as `python -m machaon.web`.

The command prints a URL with a per-launch token. Open that URL, or paste the
token into the console's token dialog. The server binds to `127.0.0.1` by
default. If `torch`, `transformers` or `safetensors` is missing, the page shows
a banner with the exact `pip install "machaon[hf]"` command instead of letting
the first session fail silently. Use `MACHAON_WEB_ALLOWED_ROOTS` to set the roots from the environment,
or repeat `--allowed-root` for more than one checkpoint directory. The
allowlist applies to checkpoint paths, edit directories, delta files and output
directories.

## Languages

The console ships in English, Simplified Chinese, Japanese and Korean. The
top-bar picker updates both static copy and dynamic job/evidence messages,
stores the choice in `localStorage`, and accepts direct links such as
`?lang=zh`, `?lang=ja` and `?lang=ko`. Without an explicit choice it follows the
browser language and falls back to English.

## First-run guide

On the first authenticated visit, the console opens a short guide that explains
the purpose of an edit, the inspect → open → baseline → propose → apply →
evaluate → save/rollback loop, and what each control does. **Start exploring**
marks the guide as seen; **Guide** in the top bar reopens it at any time. The
four-step onboarding strip remains under the hero as a compact reminder.

## First edit card

The first-authenticated page also shows a **FIRST EDIT** card. It chains the
existing pieces: choose a starter if no checkpoint is selected (SmolLM2-135M
by default, Qwen2.5-0.5B for multilingual use, or GPT-2), download it, open it,
prefill "The Eiffel Tower is in the city of -> Rome", run the teaching recipe
and report the result. The card disappears once a model session is open; the
accepted edit stays in the session for **Save**. When the recipe applies, the
card keeps a **Checkpoint output** field prefilled with `<model>-machaon` and a
**Save checkpoint** button; saving goes through the same `/api/session/save`
path as the right column. After saving, **Open saved checkpoint** closes the
edited session (rolling back its in-memory bytes), opens the saved checkpoint
and leaves a clean session for the next fact; **Done** keeps the current
session. After opening, the card evaluates the taught prompt/answer against the
reopened checkpoint and reports whether the saved continuation reproduces. It
then clears its sentence, updates the output to the new checkpoint's
`<name>-machaon`, and offers **Teach another fact**, so the same loop can
continue without returning to the main form. Each saved checkpoint starts a
fresh session and baseline; use **Teach a list** when you need one cumulative
budget and original-panel drift checks across facts. **Hide** persists the
choice in `localStorage`.

## Three-column workflow

1. **Checkpoint.** Inspect a safetensors file or a sharded checkpoint directory.
   The header summary reports tensor count, parameter count, dtype mix and shard
   count without materialising the weights. The tensor surface shows names,
   shapes and dtypes only. If you have no checkpoint, choose a starter and press
   **Download starter**: it downloads into the first allowed root with mirror
   fallback and fills the path, without uploading anything. The job banner
   streams the current endpoint, file and byte/percent progress over SSE. **Open after
   download** opens the session automatically so the next step is the
   **ONE SENTENCE** box.
2. **Edit contract.** Open a model session, capture neutral baseline prompts,
   choose an edit layer and solve a proposal. The preserve prompts become the
   null-space constraint. For quantized checkpoints, enable **Densify from
   edit layer** before a gradient solve. The proposal reports its relative
   delta and target probability before anything is written. The advanced
   **Batch edit** expander accepts multiple prompts and a JSON matrix of target
   activation rows; the resulting proposal reports exact/clamped state, target
   error, rank and a per-key error table.
3. **Evidence.** Apply the pending proposal, evaluate it against the baseline,
   and inspect the fail-closed gate verdict. The transaction timeline records
   status changes, results and errors. Rollback restores the original tensor
   bytes.

### One-sentence teaching

The **ONE SENTENCE** box at the top of the edit column is the shortest path. It
accepts the same grammar as `machaon intent` and produces the same teaching
recipe as `machaon teach`:

1. **Preview parse** compiles the sentence and fills the prompt/answer fields.
2. **Teach automatically** searches a depth window, projects the edit away from
   the preserve prompts, applies the best candidate, and keeps it only when the
   target, budget, panel and KL checks pass. A rejected candidate is rolled back
   automatically.

The optional **Also write the edited checkpoint** field saves the accepted edit
to a new directory below the allowed roots. Leave it blank to keep the edit in
the session and use the existing **Save** control later.

**Teach a list** in the same box accepts one sentence per line, runs one full
search per fact, and writes a chain with `<out>/final` and
`<out>/report.json`. The original session stays on the base checkpoint; **Use
final checkpoint** opens the last globally safe result.

Long operations run in one worker thread and are streamed as server-sent
events. The UI stays responsive while a checkpoint loads, a gradient solve runs
or a baseline is evaluated. One session owns one model at a time.

After evaluating an applied edit, use **Persist** to write the current deltas
through `Checkpoint.write_tensor` (one tensor) or `Checkpoint.write_with`
(multiple tensors). The source checkpoint remains untouched; the output
directory must be inside the allowed roots and is created as a new checkpoint.

## API surface

All routes except the static shell and `/api/health` require
`X-Machaon-Token`; the SSE route accepts the same token as a query parameter
because the browser's `EventSource` cannot set headers.

| method | route | purpose |
|---|---|---|
| GET | `/api/health` | version, roots and session status |
| GET | `/api/models` | checkpoint-looking entries below the roots |
| POST | `/api/models/fetch` | download one recommended starter below the roots |
| POST | `/api/checkpoints/describe` | header-only checkpoint summary |
| POST | `/api/checkpoints/tensors` | bounded tensor listing |
| POST | `/api/checkpoints/audit` | audit an OpenWSP v1 edit |
| POST | `/api/checkpoints/apply` | apply a `.npy` delta to one tensor |
| POST | `/api/session/open` | load a model session |
| POST | `/api/session/baseline` | capture reference distributions |
| POST | `/api/edit/propose` | solve a proposal without writing |
| POST | `/api/teach/preview` | compile a sentence into a prompt/answer contract |
| POST | `/api/teach` | run the teaching recipe on the open session |
| POST | `/api/teach_many` | run a multi-fact chain into an output container |
| POST | `/api/edit/propose_batch` | solve many prompt/value rows into one pending delta |
| POST | `/api/edit/apply` | write the pending proposal |
| POST | `/api/edit/evaluate` | score the applied edit |
| POST | `/api/edit/rollback` | restore original bytes |
| POST | `/api/wsp/audit` | classify a WSP report as repair/acquisition/non-specific |
| POST | `/api/wsp/save` | write a content-addressed WSP evidence bundle |
| POST | `/api/wsp/session` | assess the current session against matched controls, optional headroom and mechanism evidence |
| POST | `/api/wsp/session_evidence` | assess the session and save the resulting evidence bundle |
| POST | `/api/session/export_openwsp` | export the applied edit and optionally bind an evidence hash |
| POST | `/api/binding/verify` | verify an OpenWSP edit and WSP evidence bundle are bound |
| POST | `/api/session/save` | write applied deltas to a new checkpoint |
| GET | `/api/jobs/{id}` | poll a job |
| GET | `/api/jobs/{id}/events` | stream job events |

Every model-level POST returns `{"job": "...", "kind": "..."}`. The job stream
emits `status`, `result` and `error` records. A failed job keeps the browser in
a recoverable state; the session remains open and no failed write is charged to
the budget.

## Large-checkpoint example: Qwen3.6-27B

The console is model-agnostic; the same controls work for every supported
family. The 27B checkpoint is one example where the working point matters. Use
the `int4` or `int8` quantized loader on a 34 GB unified-memory host, keep the
language decoder's later layers as the edit site, and begin with a small
relative budget:

```text
path:      ./qwen36-27b
dtype:     auto
quantize:  int4
device:    cpu
layer:     60
budget β:  0.01
module:    mlp.down_proj
solver:    gradient
densify:   on
```

The checkpoint summary and tensor listing run on headers, so inspect the model
before loading it. The model guide covers the matching Transformers runtime,
the single-shard checkpoint write and the `Checkpoint.write_tensor` path:
[Qwen3.6-27B](../guides/qwen27b.md).

## Operating rules

Keep the console on loopback. Set `MACHAON_WEB_ALLOWED_ROOTS` to the smallest
directory that contains the checkpoints you edit. Treat the printed token as a
capability: it authorises a loaded model and a checkpoint write. The web layer
does not execute arbitrary Python and does not accept a free-form code field.
Use a separate process for a separate model, and save a checkpoint with the
file-level operation when you want a durable output.
