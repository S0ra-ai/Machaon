# MCP server

Machaon exposes its file-level operations over the Model Context Protocol, so an
agent can inspect a checkpoint, audit someone else's edit, apply a delta, and
gate the result — without loading a model.

```bash
pip install "machaon[mcp]"
python -m machaon.mcp_server
```

Both SDK generations work: `mcp` 1.x exposes `FastMCP`, 2.x renamed it to
`MCPServer`; the decorator and `run(transport="stdio")` surface is the same, and
`create_server()` picks whichever is installed.

CI starts the server as a subprocess and calls tools through a real client
(`scripts/mcp_smoke.py`), so a broken transport fails the build.

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

The file-level server speaks stdio and holds no model state, so it is cheap to
restart. Model-level tools share one explicitly opened `Session`.

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

## Error shape

Session tools return `{"error": "no model is open; call open_model first"}`
instead of raising, so an agent sees the reason rather than a generic
`Error executing tool`.

## Design boundaries

- **File-level tools are header-first.** They inspect, audit and rewrite
  checkpoints without loading a model into memory.
- **Model-level tools are explicit.** `open_model` is the only expensive call;
  `propose_edit` reports cost before `apply_edit` writes anything.
- **The agent chooses the site.** Layer selection belongs to the agent or to a
  procedure such as `machaon.procedures.rome`; the server enforces the
  transaction and the budget.
- **Paths use process permissions.** Run the server in the same trust boundary
  as your shell, as described in
  [SECURITY.md](https://github.com/S0ra-ai/Machaon/blob/main/SECURITY.md).

## Why file-level first

The expensive, risky operations are the ones that touch a checkpoint. Giving an
agent a verified way to inspect and audit weights is useful on its own, and it
does not require the agent to hold 3 GB of tensors in context or memory.

Model-level tools — `propose_edit`, `evaluate`, `rollback` — run on the same
session contract measured through 8B, with int8/int4 weights and a densified
local window.
