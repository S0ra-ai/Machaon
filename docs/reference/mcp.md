# MCP server

Machaon exposes its file-level operations over the Model Context Protocol, so an
agent can inspect a checkpoint, audit someone else's edit, apply a delta, and
gate the result — without loading a model.

```bash
pip install "machaon[mcp]"
python -m machaon.mcp_server
```

## Client configuration

Codex:

```bash
codex mcp add machaon -- python -m machaon.mcp_server
```

Claude Desktop (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "machaon": {
      "command": "python",
      "args": ["-m", "machaon.mcp_server"]
    }
  }
}
```

The server speaks stdio and holds no model state, so it is cheap to restart.

## Tools

| tool | what it does |
|---|---|
| `describe_checkpoint(path)` | tensor count, parameter count, dtype mix, shards — header only |
| `list_tensors(path, contains, suffix, layer, limit)` | names, shapes, dtypes; never values |
| `audit_edit(base_path, edit_dir, max_rel_norm)` | verify an OpenWSP v1 edit against the base and recompute drift |
| `apply_delta(base_path, tensor, delta_path, out_dir)` | apply one `.npy` delta to one tensor, writing a new checkpoint |
| `gate(metrics)` | apply the acceptance gates; missing metrics are failures |

Every tool delegates to a plain function in `machaon.tools`, which is what the
test suite exercises. The transport adds no behaviour.

## Model-level tools

These operate on one loaded session. `open_model` is the expensive call; every
other tool is cheap.

| tool | what it does |
|---|---|
| `open_model(path, dtype)` | load a model into the session |
| `model_status()` | parameters, layers, edited tensors, budget usage |
| `capture_baseline(neutral_prompts, corpus_texts)` | record reference distributions **before** editing |
| `propose_edit(prompt, target, layer, solver, steps, max_rel_norm)` | solve an edit and report its cost without writing |
| `apply_edit()` | write the pending proposal, charging the budget |
| `evaluate(prompts, targets)` | accuracy, KL and perplexity against the baseline |
| `rollback()` | restore every edited tensor |

The intended loop is exactly the one in `machaon.session`:

```
open_model -> capture_baseline -> propose_edit -> apply_edit -> evaluate
                                                        |
                                                    rollback
```

`propose_edit` returning the relative delta *before* anything is written is the
point. An agent can decide the edit is too expensive and never apply it.

## What it does not do

- It does not load a model, run a forward pass, or train anything.
- It does not decide *where* to edit; that belongs to the agent or to a
  procedure like `machaon.procedures.rome`.
- It is not a sandbox. Paths are read and written with the server process's
  permissions.

## Why file-level first

The expensive, risky operations are the ones that touch a checkpoint. Giving an
agent a verified way to inspect and audit weights is useful on its own, and it
does not require the agent to hold 3 GB of tensors in context or memory.

Model-level tools — `propose_edit`, `evaluate`, `rollback` — come after the
runtime is stable at 3B.
