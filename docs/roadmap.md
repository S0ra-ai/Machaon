# Roadmap

Ordered by what makes weight surgery more useful, not by what is easiest to
ship. Every completed item has a test or a measured run behind it.

## Shipped

- [x] Linear-algebra primitives with derivable invariants
- [x] Executable edit DSL: validated specs dispatch to the same pure primitives
- [x] Transactional apply with bit-exact rollback
- [x] Content-addressed delta store with verified reads
- [x] Fail-closed gates, KL quantiles, bootstrap intervals
- [x] OpenWSP v1 edit format, byte-identical, conformance-tested
- [x] Low-rank OpenWSP export with sparse bf16 correction entries
- [x] safetensors adapter: single file, directory, sharded index
- [x] HF runtime: architecture-independent layer discovery, orientation
      normalisation, padding-correct scoring
- [x] ROME procedure as a single function
- [x] Benchmark loaders with provenance (zsRE, WikiText-2, MMLU)
- [x] Long-form coherence suite, scored by the unedited model
- [x] Cumulative-edit budget with a measured cost curve
- [x] Qwen2.5 0.5B, 1.5B, 3B and 7B rungs
- [x] Llama-3.1-8B at both int8 and int4
- [x] Memory-efficient gradient solver (`densify_from`)
- [x] Six architecture families tested from configs
- [x] MCP server: six file-level tools and eleven model-level tools, tested
      over real stdio in CI
- [x] WSP acceptance protocol: matched random controls, repair-vs-acquisition
      classification, CLI and MCP audit
- [x] Reusable headroom evaluators for generation and first-token metrics
- [x] Mechanism evidence: loop, empty-output and token-limit gates with exact
      generation stop traces and matched controls
- [x] Session and MCP generation assessment that runs candidate/base/control
      generation together and enforces the mechanism gate
- [x] Labelled obstruction-vs-capability benchmark harness with explicit
      synthetic fixtures and a confusion-matrix summary
- [x] Evidence-driven planning: edit / diagnose / do-not-edit routes, layer
      ranking from causal-patching scores, and no guessed edit site
- [x] Live layer-selection search: apply, evaluate and roll back each candidate
      through the session contract, keeping per-trial evidence
- [x] One-sentence intent compiler and one-call teaching recipe: automatic
      baseline, depth-window search, target/panel/KL checks and bit-exact
      rollback
- [x] Agent-callable ROME layer search over MCP with automatic rollback
- [x] Conservative automatic diagnosis: target headroom plus mechanism relief
      for obstruction; low headroom without relief for capability gap
- [x] Measured/synthetic provenance gate for WSP benchmark cases
- [x] Unit symmetry alignment for permutation and positive-scaling changes
- [x] Permutation-aware linear MLP merge after alignment
- [x] Probe-based functional verification of aligned MLP units
- [x] Task-vector extraction and signed, budgeted combinations
- [x] Deterministic weight fingerprints and sparse edit-diff reports
- [x] Measured threshold-classifier reference case for provenance-gate smoke
- [x] Content-addressed WSP evidence bundles with tamper verification
- [x] Local browser console over FastAPI/SSE: token auth, path allowlist,
      checkpoint inspection, session editing, evidence and rollback
- [x] Documentation site in English, Chinese, Japanese and Korean, deployed to
      GitHub Pages
- [x] Release workflow: sdist, wheel, GitHub release, optional PyPI publish

## Now

- [x] **MEMIT-style batch solving.** Many facts, one constraint system, one
      budget; exact, protected and clamped outcomes are reported explicitly.
- [x] **L1 action space: constrained Python edit functions.** A declared
      `def edit(W): ...` over one selected tensor runs in a resource-capped
      worker under an AST allowlist, then passes the same budget and
      protected-key contract before session apply / rollback.
- [ ] **More measured rungs.** A 13B/14B checkpoint path and a reproducible
      memory/throughput table.
- [ ] **Deeper long-form suite.** Multi-turn prompts, entity consistency and
      contradiction detection across 1k+ tokens.

## Next

- [ ] Larger measured benchmark coverage for the obstruction-vs-capability harness
- [ ] A public leaderboard of reproducible edit runs
- [ ] Native Apple Silicon and CUDA benchmark matrix

## Adjacent tools

Machaon's unit is a targeted edit to an existing model. Nearby jobs have mature
tools, and their artefacts compose with this one:

- **Merging and task arithmetic:** mergekit deltas audit and apply through
  `machaon.tools` and `machaon.verify`.
- **Pruning and rewiring:** structural edits arrive as deltas and pass the same
  budget and gate checks.
- **Fine-tuning and LoRA:** an equal-budget LoRA baseline is measured in
  [`RESULTS.md`](RESULTS.md#r9-equal-budget-lora-baseline), and adapter deltas
  can be compared with rank-one edits.
- **Serving:** export the edited checkpoint with `Checkpoint.write_with` and
  serve it with any Hugging Face-compatible stack.
