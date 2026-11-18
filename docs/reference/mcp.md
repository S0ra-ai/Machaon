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
| `wsp_audit(report)` | classify a WSP report as repair, acquisition, non-specific, or no-effect |
| `wsp_save(report, out_dir, metadata)` | write a content-addressed WSP evidence bundle |
| `verify_binding(edit_dir, evidence_dir)` | verify an OpenWSP edit and evidence bundle are bound |
| `plan_surgery(failure_type, layer_scores, patch_report, module_suffix, max_rel_norm, min_score)` | choose edit / diagnose / do-not-edit from failure and site evidence |
| `diagnose_failure(base, headroom, mechanism, target_metric, min_target_gap, min_mechanism_gap)` | classify obstruction vs capability gap without editing |
| `parse_intent(sentence)` | compile a plain-language teaching request into an exact contract |
| `check_saved(model, sentence, prompt, answer, facts, dtype, device, quantize)` | reload a saved checkpoint and evaluate one fact or a fact list without a shared session |
| `plan_program(path)` | parse a portable `.machaon` text/JSON program and return model/output/download preflight without loading a model |
| `run_program(path, verify, out, report_out)` | run a portable program with optional output/report overrides and saved-byte verification |
| `verify_program_report(path, program_path, check_model, checkpoint)` | verify a report's content address, optional program hash, and optional saved-byte facts |

Header and evidence tools delegate to plain functions in `machaon.tools`.
`check_saved` delegates to `machaon.check`, loads weights but never opens or
changes the shared session. `plan_program` is header-only and reports the same
model, output, report-path and starter-download preflight as the CLI/Web plan;
`run_program` delegates to the same program runner as the CLI and Web console,
including its saved-byte verification. `verify_program_report` returns
`{"valid": false, "error": ...}` for a malformed or modified report instead of
raising through the transport. The server exposes 28 tools. The transport adds
no behaviour.

## Model-level tools

These operate on one loaded session. `open_model` is the expensive call; every
other tool is cheap.

| tool | what it does |
|---|---|
| `open_model(path, dtype)` | load a model into the session |
| `model_status()` | parameters, layers, edited tensors, budget usage |
| `capture_baseline(neutral_prompts, corpus_texts)` | record reference distributions **before** editing |
| `propose_edit(prompt, target, layer, solver, steps, max_rel_norm)` | solve an edit and report its cost without writing |
| `propose_batch(prompts, values, layer, preserve_prompts, max_rel_norm, clamp, ridge)` | solve many rows into one pending batch delta |
| `propose_action(source, intent, layer, keys, values, preserve, max_rel_norm, timeout_seconds)` | compile a constrained `def edit(W): ...` into a pending delta |
| `apply_edit()` | write the pending proposal, charging the budget |
| `evaluate(prompts, targets)` | accuracy, KL and perplexity against the baseline |
| `assess_wsp(prompts, targets, headroom, mechanism, seed)` | evaluate the edit plus a matched control, optional obstruction gate, and classify it |
| `assess_generation_wsp(prompts, expected, max_new_tokens, headroom_max_new_tokens, match, seed)` | generate candidate/base/control/headroom and attach the mechanism gate |
| `search_rome_layers(prompt, target, layers, solver, max_rel_norm, metric, max_evaluations)` | try ROME at each layer, evaluate and roll back |
| `teach(sentence, prompt, answer, layers, solver, max_rel_norm, dry_run)` | run the full teaching recipe: search, apply, verify, or roll back |
| `rollback()` | restore every edited tensor |

The intended loop is exactly the one in `machaon.session`:

```
open_model -> capture_baseline -> propose_edit -> apply_edit -> evaluate
                                                        |
                                                    rollback
```

`propose_edit` returning the relative delta *before* anything is written is the
point. An agent can decide the edit is too expensive and never apply it.

`teach` is the intent-level shortcut. It accepts a `sentence` (compiled by
`parse_intent`) or an explicit `prompt` / `answer`, chooses a depth window when
`layers` is omitted, applies at most one accepted edit, and rolls back on a
failed local check. The returned report keeps every trial. Call `rollback` to
undo an applied teaching edit.

`propose_action` accepts a small, declared Python function rather than
arbitrary code. Its AST allowlist rejects imports, loops, private attributes
and arbitrary calls before a resource-capped worker runs it. The returned delta
still has to pass the shape, budget and protected-key checks, and nothing is
written until `apply_edit`.

## Error shape

Session tools return `{"error": "no model is open; call open_model first"}`
instead of raising, so an agent sees the reason rather than a generic
`Error executing tool`.

## Design boundaries

- **File-level tools are header-first.** They inspect, audit and rewrite
  checkpoints without loading a model into memory; `check_saved` is the
  explicit exception that reloads a checkpoint for a no-session fact check.
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
