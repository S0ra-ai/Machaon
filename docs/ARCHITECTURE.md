# Architecture

Machaon is three layers that know almost nothing about each other.

```
                    ┌──────────────────────────┐
   edit spec  ───►  │  machaon.linalg          │  pure NumPy, no model
   (DSL, JSON)      │  rank1 · null-space · ...│
                    └───────────┬──────────────┘
                                │ delta (ndarray)
                    ┌───────────▼──────────────┐
                    │  machaon.transaction     │  bit-exact rollback
                    │  machaon.store           │  content-addressed
                    └───────────┬──────────────┘
                                │ applied / rejected
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
┌───────▼───────┐     ┌─────────▼─────────┐   ┌─────────▼─────────┐
│ machaon.runtime│     │ machaon.hygieia   │   │ machaon.interop   │
│ HF / safetensors│    │ gates · runner    │   │ OpenWSP v1        │
└───────────────┘     └───────────────────┘   └───────────────────┘
```

## The human layer

`machaon.intent` compiles a small, explicit language into an exact
`(prompt, answer)` pair. `machaon.recipes.teach_fact` is the loop above the
primitives: it captures an unrelated baseline panel, searches candidate layers
through `Session.search_rome_layers`, projects each trial away from the panel,
applies the winner, checks the target, panel and KL, and rolls back on failure.
The CLI, the MCP `teach` tool and the Web **ONE SENTENCE** box all call this one
function; none of them reimplement the contract.

`machaon.program` is the shareable form of the same loop. A small text or JSON
file declares the model, facts, protected prompts, budget and runtime options;
the runner resolves relative paths, calls `teach_many`, writes a program
report, and reloads the saved checkpoint through `check_facts` before it is
accepted. The program is an orchestration record, not a second edit engine.

## The two contracts

**Runtime.** `next_logits(items) -> (n, vocab)` and `nll(items) -> (n,)`.
That is the entire model-facing surface. `HFModel` implements it over a
Hugging Face causal LM; a test backend implements it in four lines.

**Edit.** `(W, k, v, preserve) -> (W_new, delta)`. Pure arrays. No framework
types, no device, no dtype opinions: everything internal is float64.

Because the contracts are this narrow, adding an architecture means adding a
`LAYER_PATHS` entry, and adding an edit means adding one pure function.

## Why the maths layer is NumPy

- it is testable on any machine, including CI without torch;
- a delta is portable across frameworks;
- the guarantee (`ΔW k_i = 0`) is a statement about numbers, not a framework.

Torch appears only in `runtime/` and `adapters/`, at the boundary.

## Why transactions rather than in-place edits

Rollback is part of the edit. `Transaction` copies the original bytes before
applying, so `rollback()` restores them exactly. `commit(verdict)` refuses when
the verdict did not pass, which makes the default fail-closed.

## Why the gates fail closed

A missing metric is a failure. Otherwise the easiest way to pass the suite is to
stop measuring, and every tool that makes measurement optional eventually is
used without it.

`machaon.wsp` is the decision layer above the gates. It asks a narrower
question than "did the target metric move?": given the base score, the
candidate score, a same-support/same-norm random control, and an optional
headroom baseline, is the edit a *repair* of an existing capability or an
*acquisition* of a new one? A positive result without a headroom baseline is
reported as unclassified, and a matched control that reaches the candidate
rejects the effect as non-specific. The decision is executable from Python,
the `machaon wsp` CLI, and the `wsp_audit` MCP tool.

`machaon.mechanism` adds the obstruction side of that decision. Generation
traces yield loop rate, empty-output rate and exact token-limit hits; a
mechanism report compares those metrics with the unedited base and a matched
control. When a mechanism payload is supplied to WSP, a target gain that leaves
the obstruction in place is rejected rather than reported as a successful
repair.

`machaon.executor` is the other side of the DSL. The JSON edit spec is not just
an audit record: `execute_spec` dispatches each validated op to the same pure
NumPy primitive, refuses `solver="search"`, and rejects a result that exceeds
the spec's declared relative-norm budget. The executor never writes a
checkpoint and never trains.

`machaon.action` is the L1 action space for agent-written edit functions. A
`def edit(W): ...` source is checked against a strict AST allowlist, compiled in
a separate worker process with a wall-clock deadline and resource caps, and
given only the selected tensor plus optional `keys`, `values` and `preserve`
arrays. The parent rejects imports, loops, private attributes, arbitrary calls,
wrong shapes, non-finite values, over-budget deltas and protected-key drift.
`Session.propose_action` places the surviving delta on the same pending /
apply / rollback path as every other edit.

`machaon.planning` is the decision layer above diagnosis and below execution.
It consumes failure labels and layer-site scores, then emits `edit`,
`diagnose` or `do_not_edit`. A positive obstruction label with no site above
threshold still routes to diagnosis; a capability gap never becomes a weight
edit by default.

`Session.search_layers` closes the loop when static site scores are absent: for
each candidate it creates a pending proposal, applies it, evaluates the model,
and rolls back in a `finally` path. Every trial remains in the report; cleanup
failure aborts the search instead of continuing from altered weights.

`diagnose_failure` classifies pre-edit evidence conservatively. It calls an
obstruction only when the target headroom and at least one mechanism metric
improve together; a low headroom without mechanism relief becomes a capability
gap; missing or contradictory evidence stays unknown.

## Format boundary

`machaon.interop.openwsp` is a self-contained implementation of an external
format. It is conformance-tested against an artefact produced by the reference
implementation, not against a description of it. If the address changes upstream,
the test fails rather than silently drifting.

## Design boundaries

- **Closed-form first.** Model weights are never optimised by gradient descent;
  a ROME solve optimises one value vector, then hands the delta to the maths.
- **Text at the boundary.** Text enters through a runtime's tokenizer; the edit
  layer only ever sees arrays.
- **Orchestration by the caller.** `Session` and the MCP server provide the
  state machine — open → baseline → propose → apply → evaluate → rollback —
  and leave scheduling to the agent or script that owns the job.
