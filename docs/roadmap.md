# Roadmap

Ordered by what would make the project harder to ignore, not by what is easiest.

## Done

- [x] Linear-algebra primitives with derivable invariants
- [x] Transactional apply with bit-exact rollback
- [x] Content-addressed delta store and snapshots
- [x] Fail-closed gates, KL quantiles, bootstrap intervals
- [x] OpenWSP v1 edit format, byte-identical, conformance-tested
- [x] safetensors adapter: single file, directory, sharded index
- [x] HF runtime: architecture-independent layer discovery, orientation
      normalisation, padding-correct scoring
- [x] ROME procedure as a single function
- [x] Benchmark loaders with provenance (zsRE, WikiText-2, MMLU)
- [x] First real result on GPT-2 124M

## Next

- [ ] **Long-form coherence suite.** The literature says this is where edits
      break. It is the highest-value missing measurement.
- [ ] **Cumulative-edit budget.** Cap `Σ‖ΔWᵢ‖/‖W‖` and show where the model
      collapses.
- [x] **0.5B rung.** Qwen2.5-0.5B: `model.layers`, bf16 I/O, same benchmark
      panel. Results in [`reference/benchmarks.md`](reference/benchmarks.md).
- [ ] **1.5B and 3B rungs.** Memory and shard handling.
- [ ] **Up to 8B.** Single-shard edits without loading the whole model.

## Later

- [ ] MCP server: `propose_edit`, `apply_edit`, `evaluate`, `diff`, `rollback`
- [ ] L1 action space: sandboxed Python edit functions
- [ ] MEMIT-style batch solving
- [ ] LoRA baseline at equal compute
- [ ] Docs deployment to GitHub Pages

## Explicitly out of scope

- Training or fine-tuning
- Models above 8B
- Serving inference
- Prompt optimisation
