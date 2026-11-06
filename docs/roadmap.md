# Roadmap

Ordered by what makes weight surgery more useful, not by what is easiest to
ship. Every completed item has a test or a measured run behind it.

## Shipped

- [x] Linear-algebra primitives with derivable invariants
- [x] Transactional apply with bit-exact rollback
- [x] Content-addressed delta store with verified reads
- [x] Fail-closed gates, KL quantiles, bootstrap intervals
- [x] OpenWSP v1 edit format, byte-identical, conformance-tested
- [x] safetensors adapter: single file, directory, sharded index
- [x] HF runtime: architecture-independent layer discovery, orientation
      normalisation, padding-correct scoring
- [x] ROME procedure as a single function
- [x] Benchmark loaders with provenance (zsRE, WikiText-2, MMLU)
- [x] Long-form coherence suite, scored by the unedited model
- [x] Cumulative-edit budget with a measured damage curve
- [x] Qwen2.5 0.5B, 1.5B, 3B and 7B rungs
- [x] Llama-3.1-8B at both int8 and int4
- [x] Memory-efficient gradient solver (`densify_from`)
- [x] Six architecture families tested from configs
- [x] MCP server: five file-level tools and seven model-level tools, tested
      over real stdio in CI
- [x] Documentation site in English, Chinese, Japanese and Korean, deployed to
      GitHub Pages
- [x] Release workflow: sdist, wheel, GitHub release, optional PyPI publish

## Now

- [ ] **MEMIT-style batch solving.** Many facts, one constraint system, one
      budget. The single-edit path is the building block.
- [ ] **L1 action space: sandboxed Python edit functions.** A declared function
      over tensors, executed with the same transaction and gate contract.
- [ ] **More measured rungs.** A 13B/14B checkpoint path and a reproducible
      memory/throughput table.
- [ ] **Deeper long-form suite.** Multi-turn prompts, entity consistency and
      contradiction detection across 1k+ tokens.

## Next

- [ ] Layer-selection search driven by the evaluation harness
- [ ] Sparse and structured edit forms in the OpenWSP writer
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
