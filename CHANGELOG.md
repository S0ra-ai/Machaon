# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).

## [0.3.3] - 2026-11-08

### Added

- `examples/contract_synthesis.py`: a zero-download demonstration of
  budget-aware protected-set admission and post-hoc protected-key auditing.
- Robust per-file endpoint fallback in `scripts/fetch_model.py`. If the endpoint
  that listed a repository cannot serve a file, the downloader tries the
  remaining endpoints and reports each failure.
- Content-length verification for downloads; incomplete `.part` files are
  removed before the next endpoint is tried.
- Four download-fallback tests and an `.DS_Store` ignore rule.

## [0.3.4] - 2026-11-08

### Added

- Qwen3.5/Qwen3.6 multimodal layer discovery through
  `model.language_model.layers`, plus a loader that selects
  `AutoModelForImageTextToText` for conditional-generation checkpoints.
- A four-language Qwen3.6-27B guide covering int4 evaluation, local dense
  windows, checkpoint-only shard writes, and the session workflow.
- A multimodal-wrapper architecture test.

### Changed

- Repository test count badge and installation examples now report the current
  suite size.

## [0.3.2] - 2026-11-06

### Added

- `machaon.contract.synthesize_contract`: budget-aware protected-set synthesis.
  It admits the largest auditable subset under a relative-norm budget, keeps
  the target and every admitted key exact, and returns rejected keys explicitly.
- Six contract tests covering full admission, partial coverage, budget refusal,
  weighted priorities and invalid inputs.
- API documentation and translated notes for contract synthesis.

### Changed

- README, maths docs and API reference now distinguish budget clamping from
  budget-aware coverage.

## [0.3.1] - 2026-11-06

### Added

- `examples/README.md`: a task-oriented index of every runnable example, with
  the exact command and the measurement each one produces.
- A support-matrix table in the README that separates measured checkpoints from
  config-level architecture coverage.
- A community section that points questions, bugs, security reports and changes
  at the right channel.

### Changed

- README and concept pages now describe locality, reversibility and acceptance
  as positive operating properties, with the measured envelope stated directly.
- The quickstart demo output is synchronized with the current deterministic run;
  the README notes the 1e-16 BLAS variation instead of hiding it.
- Benchmark prose separates sample-size effects from capacity effects and
  labels small MMLU panels as smoke checks.

### Fixed

- Enabled GitHub Discussions for the support link already advertised by the
  issue-template configuration.
- Refreshed the demo drift values across the English, Chinese, Japanese and
  Korean quickstarts.

## [0.3.0] - 2026-11-06

### Added

- `machaon doctor` (human and `--json`), `python -m machaon`, `--json` on
  `inspect`, and a `machaon-mcp` console script.
- Python 3.13 in the test matrix, a branch-coverage job at a 78% floor with
  **80% measured**, a real MCP stdio job, a wheel/sdist build-and-install job,
  CodeQL, Dependabot, `pre-commit`, `CODEOWNERS` and `SUPPORT.md`.
- `mkdocs-static-i18n` as a declared docs dependency, translated navigation and
  complete Chinese, Japanese and Korean editions of every documentation page.
- `docs/comparison.md`: where Machaon sits relative to fine-tuning, LoRA,
  ROME/MEMIT, mergekit, pruning and activation steering.
- Delta-store integrity checks: reads verify the SHA-256 address by default,
  writes publish atomically, and `StoreCorrupted` distinguishes corruption
  from a missing file.
- Tests for CLI exit codes, `python -m machaon`, store corruption, non-finite
  inputs, version consistency, packaging metadata and documentation tone.

### Changed

- `Session.open` honours `float32`, `float16`, `bfloat16` and `auto`, and
  supports an explicit device. Budget is checked before the write and recorded
  only after it succeeds.
- The 3B benchmark row now reports the measured layer-34 run (rephrase 0.327),
  consistent with `docs/RESULTS.md` and the README.
- Documentation describes capabilities first; scope is stated as a design
  boundary with the adjacent tool named.

### Fixed

- The docs workflow now installs the i18n plugin it configures, so
  `mkdocs build --strict` succeeds instead of failing on an unknown plugin.
- The `mcp` extra carries a Python 3.10+ environment marker; `.[mcp]` resolves
  on Python 3.9 without pulling an incompatible SDK.
- CI now runs the MCP smoke test it advertises, and exercises the CLI entry
  points in the package job.
- The 202-test suite is green on Python 3.9 and 3.12; lint passes on
  `machaon`, `tests`, `examples` and `scripts`.

## [0.2.0] - 2026-11-04

### Added

- 7B evaluation on a 32 GB machine: int8 weights plus `densify_from(layer)`,
  so backward traverses only the blocks after the edit. Rewrite 1.000,
  rephrase 0.929, perplexity 1.007 at rel.Δ 0.0043.
- `--lr` and `--quantize int4` on the benchmark script.
- Llama-3.1-8B (8.03B) evaluated at both int8 and int4; documented the
  efficacy/damage trade-off at the memory boundary.
- Equal-budget LoRA baseline (`examples/lora_baseline.py`): rank-8 reaches
  1.000/1.000 but costs 44% perplexity and 25 MMLU points, against 1.000/0.976
  for 4% with the rank-one edit.
- Matched 3B/7B comparison (3 targets, 42 rephrasings): 3B 1.000/0.857 vs 7B
  1.000/0.929, correcting the earlier "capacity ceiling" reading.
- 3B depth sweep (layers 26/32/34): rephrase 0.230 → 0.327 as the edit moves
  closer to the output; rewrite stays at 0.625, which is a capacity limit.
- `--quantize int8` on the benchmark script, with the memory preflight adjusted.
- `scripts/mcp_smoke.py`: starts the MCP server over stdio and calls tools with
  a real client; wired into CI as its own job.
- `create_server()` supports both `mcp` 1.x (`FastMCP`) and 2.x (`MCPServer`).
- Session tools return a structured error instead of raising.
- Fixed `pyproject.toml`: `keywords` and `dependencies` had drifted under
  `[project.urls]`, which broke every `pip install .`.
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
