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
default. Use `MACHAON_WEB_ALLOWED_ROOTS` to set the roots from the environment,
or repeat `--allowed-root` for more than one checkpoint directory. The
allowlist applies to checkpoint paths, edit directories, delta files and output
directories.

## Three-column workflow

1. **Checkpoint.** Inspect a safetensors file or a sharded checkpoint directory.
   The header summary reports tensor count, parameter count, dtype mix and shard
   count without materialising the weights. The tensor surface shows names,
   shapes and dtypes only.
2. **Edit contract.** Open a model session, capture neutral baseline prompts,
   choose an edit layer and solve a proposal. The preserve prompts become the
   null-space constraint. For quantized checkpoints, enable **Densify from
   edit layer** before a gradient solve. The proposal reports its relative
   delta and target probability before anything is written.
3. **Evidence.** Apply the pending proposal, evaluate it against the baseline,
   and inspect the fail-closed gate verdict. The transaction timeline records
   status changes, results and errors. Rollback restores the original tensor
   bytes.

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
| POST | `/api/checkpoints/describe` | header-only checkpoint summary |
| POST | `/api/checkpoints/tensors` | bounded tensor listing |
| POST | `/api/checkpoints/audit` | audit an OpenWSP v1 edit |
| POST | `/api/checkpoints/apply` | apply a `.npy` delta to one tensor |
| POST | `/api/session/open` | load a model session |
| POST | `/api/session/baseline` | capture reference distributions |
| POST | `/api/edit/propose` | solve a proposal without writing |
| POST | `/api/edit/apply` | write the pending proposal |
| POST | `/api/edit/evaluate` | score the applied edit |
| POST | `/api/edit/rollback` | restore original bytes |
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
