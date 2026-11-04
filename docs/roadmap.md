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

- [x] **Long-form coherence suite.** Reference-scored by the unedited model,
      repetition reported as a delta against baseline.
- [x] **Cumulative-edit budget.** `EditBudget` caps `Σ‖ΔWᵢ‖/‖W‖`, is charged at
      commit, and is validated by a measured damage curve.
- [x] **0.5B rung.** Qwen2.5-0.5B: `model.layers`, bf16 I/O, same benchmark
      panel. Results in [`reference/benchmarks.md`](reference/benchmarks.md).
- [x] **1.5B rung.** Qwen2.5-1.5B, same panel; file-level write at 2.9 GB.
- [x] **3B rung.** Sharded checkpoint, 11.7 GB peak RSS, same panel.
- [x] **Up to 8B at the checkpoint level.** Single-shard edits without
      loading the whole model.
- [ ] **Up to 8B at the evaluation level.** Blocked by 32 GB of RAM;
      `check_fit` refuses instead of thrashing.

- [x] **Six architecture families** tested from configs (GPT-2, Llama, Mistral,
      Gemma, Qwen3, Phi-3).

## Later

- [x] MCP server (file-level): `describe_checkpoint`, `list_tensors`,
      `audit_edit`, `apply_delta`, `gate`
- [x] MCP server (model-level): `open_model`, `capture_baseline`,
      `propose_edit`, `apply_edit`, `evaluate`, `rollback`
- [ ] L1 action space: sandboxed Python edit functions
- [ ] MEMIT-style batch solving
- [ ] LoRA baseline at equal compute
- [ ] Docs deployment to GitHub Pages

## Explicitly out of scope

- Training or fine-tuning
- Models above 8B
- Serving inference
- Prompt optimisation
