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

An edit that cannot be undone is not an edit; it is a deployment. `Transaction`
copies the original bytes before applying, so `rollback()` restores them
exactly. `commit(verdict)` refuses when the verdict did not pass, which makes the
default fail-closed.

## Why the gates fail closed

A missing metric is a failure. Otherwise the easiest way to pass the suite is to
stop measuring, and every tool that makes measurement optional eventually is
used without it.

## Format boundary

`machaon.interop.openwsp` is a self-contained implementation of an external
format. It is conformance-tested against an artefact produced by the reference
implementation, not against a description of it. If the address changes upstream,
the test fails rather than silently drifting.

## What is deliberately absent

- **No optimiser over model weights.** There is no gradient descent here.
- **No prompt handling.** Text enters only through a runtime's tokenizer.
- **No scheduler.** Batch orchestration belongs to the caller or to a future
  MCP server.
