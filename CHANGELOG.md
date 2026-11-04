# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- MCP contract test: a recording FastMCP checks every tool is registered,
  documented and delegating, without the SDK installed.
- Six architecture families tested from tiny configs; supported-families table
  in the API reference.
- `machaon.memory.check_fit`: refuse a checkpoint whose weights exceed 60% of
  available memory, with the measured rationale.
- `--force` on the benchmark script to override the preflight.
- `Checkpoint.write_tensor`: rewrite one shard and hard-link the rest. Verified
  on Qwen2.5-7B: 4 shards, 2.3 s, ~4 GB peak.
- `examples/file_edit.py --single-shard`.
- 7B memory wall documented: checkpoint edits work, full evaluation does not fit
  in 32 GB.
- `examples/longform_check.py` takes `--model/--layer/--steps/--tokens`; first
  cross-architecture long-form numbers (1.5B: repetition Δ +0.099, NLL 0.938).
- `machaon.session.Session`: baseline → propose → apply → evaluate → rollback,
  with the budget charged at apply.
- MCP model-level tools over one session.
- `solve_edit` separated from `rome_edit`, so a proposal can be inspected
  before it is written.
- Qwen2.5-3B rung: sharded checkpoint, mean rel.delta 0.0052, rewrite 0.625,
  WikiText-2 1.002, MMLU unchanged.
- MCP stdio server (`machaon.mcp_server`) with five file-level tools, and
  `machaon.tools` as their transport-free implementation.
- Qwen2.5-1.5B rung: same benchmark panel, half the relative delta and 0.5%
  perplexity cost; file-level checkpoint write in 2.3 s.
- `examples/file_edit.py`: rewrite a checkpoint on disk without loading the
  model twice.
- `scripts/fetch_model.py`: endpoint fallback (Hugging Face → mirror →
  ModelScope) with safetensors header verification.
- `Checkpoint.summary()`: tensor/parameter counts without loading weights.
- Long-form coherence checker (`machaon.hygieia.longform`) with
  reference-scored NLL ratio and repetition delta; no judge model.
- `HFModel.generate` (greedy, deterministic).
- `examples/longform_check.py`.
- `machaon.budget.EditBudget`: cumulative relative-change budget, charged at
  commit, with per-tensor limits.
- `Workspace(budget=...)`: automatic registration and atomic refusal.
- `examples/budget_sweep.py`: measured damage curve for 0/8/16/32 edits.
- Qwen2.5-0.5B support: `model.layers`, bf16/float32 I/O, same benchmark panel.
- Target-injection benchmark protocol, comparable across model strengths.
- `solver` choice in `rome_edit`: `gradient` (smaller deltas) and `direction`
  (no backward pass, larger deltas).
- Documentation site (Material for MkDocs): getting started, concepts, guides,
  API and CLI reference, benchmarks, FAQ, roadmap.
- Open-benchmark evaluation: zsRE, WikiText-2 and MMLU loaders with recorded
  provenance (source, revision, licence, SHA-256).
- `machaon.procedures.rome` and `machaon.runtime.HFModel.score_continuations`.
- `docs/reference/benchmarks.md` with licences, protocol and current numbers.
- First real result on GPT-2 124M (`examples/gpt2_rome.py`, `docs/RESULTS.md`):
  rewrite 1.00, paraphrase 1.00, KL p95 0.0006, perplexity ratio 1.0008.
- `paraphrase_acc` in the evaluation runner.
- `machaon.runtime.HFModel`: architecture-independent layer discovery, float64
  reads, dtype-preserving writes, next-token logits, token NLL, and
  `capture_module_input` for ROME-style keys.
- `machaon.hygieia.runner`: a backend-agnostic evaluation runner
  (`EditSuite`, `evaluate_edit`, `judge`) reporting efficacy, KL and perplexity
  ratio.
- `examples/tiny_edit.py`: a no-download end-to-end demonstration where the same
  edit passes one budget and fails another.
- `machaon.adapters.Checkpoint`: safetensors reading and rewriting for single
  files, directories, and sharded indexes.
- `machaon.interop.openwsp`: byte-identical OpenWSP v1 edit format.
- `machaon.refusals` and `machaon.verify`: pre-acceptance rules and post-hoc
  protected-key auditing.
- `scripts/ci.sh`, issue templates, and contribution docs.

### Fixed

- `Conv1D` weights (GPT-2) are read and written in `(out, in)` orientation.
- Tokenizers without a pad token: manual padding plus attention masks.
- Padded batches now score the last real token, not the pad token.

## [0.0.1] - 2026-11-03

### Added

- Linear-algebra primitives: rank-one, null-space constrained rank-one,
  direction ablation and injection, low-rank truncation.
- Declarative edit DSL with a mandatory `intent`.
- Transactional apply with bit-exact rollback.
- Content-addressed delta store and workspace snapshots.
- Fail-closed acceptance gates with KL, perplexity and bootstrap intervals.
- Activation-patching scaffold.
