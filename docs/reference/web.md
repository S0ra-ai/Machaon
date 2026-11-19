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
machaon web --allowed-root /models --port 8765
# machaon-web is the same launcher
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

## Interface

The console is one page. A checkpoint bar sits at the top, followed by the
**Teach a fact** composer and four native disclosures: **Model**, **Edit**,
**Weight program** and **Results**. There is no first-run modal or separate
hero; the workflow is visible in the order the controls appear.

### Checkpoint bar

Paste a checkpoint directory or a single `.safetensors` file and press
**Inspect**. **Browse** lists checkpoint directories found below the configured
allowed roots. The path is remembered in `localStorage` after a successful open
or save. The checkpoint bar stays available while the rest of the page changes.

### Teach a fact

This is the primary path. Enter one sentence in the form:

```text
The Eiffel Tower is in the city of -> Rome
```

Choose a starter when no checkpoint is selected, optionally choose CPU, auto,
Apple MPS or CUDA, and press **Teach**. The card parses the sentence through the
same grammar as `machaon intent`, opens the selected checkpoint or downloads the
starter when necessary, applies the teaching recipe, and shows the same target,
budget, panel and KL evidence as the CLI. A rejected edit is rolled back.

After an apply, the output field and **Save** action appear. Saving writes a new
checkpoint; **Open saved** closes the edited session, opens the saved bytes and
rechecks the taught answer before clearing the sentence for another fact. A
session that is already open can be taught directly from the same card; starter
and device selectors disappear when they no longer affect the run.

**Teach a list** starts the cumulative multi-fact chain. **Create program**
turns the current fact into an editable `.machaon` program. Both actions remain
available after a successful edit.

### Model

The **Model** disclosure contains header-only inspection, tensor names, shapes
and dtypes, optional starter download, session dtype/device/budget settings, baseline capture, and one-fact
verification without editing. A starter download streams its endpoint,
file and byte progress through SSE. **Open after download** opens the session
when the download finishes.

The **Verify a fact** subsection reloads the selected checkpoint, evaluates one
prompt/answer or one fact per line, and reports PASS/FAIL with each answer and
accuracy. It calls `/api/check` and never edits the session.

### Edit

The **Edit** disclosure is the exact contract editor. It captures a prompt,
target and preserve prompts, then handles proposal, apply, evaluate and rollback. A proposal is a dry run; **Apply** writes inside the transaction and
charges the cumulative budget. A failed gate or rollback restores the original
bytes.

**Advanced edit** exposes layer, solver, step, learning-rate, module-suffix,
relative-budget and densify settings. **Teach a list** runs one search per fact
with a cumulative budget and original-panel drift check. **Batch edit** accepts
multiple prompts and activation rows and reports exact/clamped state, error,
rank and per-key error.

### Weight program

The **Weight program** disclosure accepts the same `.machaon` text or JSON as
`machaon program`. **Check** calls `/api/program/plan` and resolves paths, facts,
budget, device and output without loading a model. **Run** executes the chain as
a background job, writes `<output>/program-report.json`, reloads the saved
checkpoint and verifies every fact before accepting the run.

**More** contains the program folder and report path, current-checkpoint
prefill, local `.machaon` import, SmolLM2 / two-fact / bilingual-Qwen templates,
recent programs and runs, the fact-list generator, and report verification.
**Verify report** checks the report's SHA-256 and structure; **Verify + model**
also reloads the checkpoint and re-runs the facts. Optional checkboxes require
the current editor program to match or use the selected checkpoint instead of
the recorded path. Relative paths and all browser file access use the same
allowlist as every other console operation.

### Results

The **Results** disclosure contains the latest proposal, fail-closed gate
verdict, checkpoint saving, WSP controls and the transaction activity log. WSP
supports report audit, session assessment, evidence save, OpenWSP export and
binding verification. Missing gate metrics still fail closed.

### Local state and languages

The console ships in English, Simplified Chinese, Japanese and Korean. The
top-bar picker updates static copy and dynamic job/evidence messages, stores the
choice in `localStorage`, and accepts direct links such as `?lang=zh`,
`?lang=ja` and `?lang=ko`. Recent programs and runs are local-only history in
`localStorage`; they are never uploaded.

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
| POST | `/api/check` | evaluate one fact against a saved checkpoint |
| POST | `/api/program/plan` | parse a pasted weight program without loading a model |
| POST | `/api/program/run` | run a pasted weight program and verify the saved checkpoint |
| POST | `/api/program/verify` | verify a report's SHA-256, optional program match, and optional saved-byte facts |
| POST | `/api/program/new` | generate the shared `.machaon` text from facts and paths |
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
