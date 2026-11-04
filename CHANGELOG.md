# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

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
