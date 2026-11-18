# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `machaon program-new`: generate a reviewable `.machaon` text program from one
  or more `--teach` sentences. The generated file is round-tripped through the
  normal program parser, defaults to the SmolLM2 starter and a derived output
  path, supports custom model/output/fetch/budget/policy values, writes to
  standard output with `-`, and refuses to overwrite without `--force`.
- `machaon.program.render_program_text(...)` exposes the same generator to
  Python callers, including the advanced program fields.
- `machaon.program` and `machaon program`: a portable text or JSON weight
  program. One file declares the model, facts, protected prompts, cumulative
  budget and runtime policy; `--plan` resolves the contract without loading a
  model, and a run delegates to `teach-many`, writes
  `<output>/program-report.json`, then reloads the saved checkpoint and checks
  every fact before accepting the run. Relative paths resolve from the program
  file. The Web console exposes the same text/JSON program through
  `/api/program/plan` and `/api/program/run`, constrained by the existing
  filesystem allowlist; it can prefill a program from the current checkpoint
  and download the editor text as a `.machaon` file. The console also imports
  local `.machaon` files, starts from one- and two-fact SmolLM2 templates, and
  remembers the last eight programs in `localStorage`.
- MCP `plan_program` and `run_program` expose the portable program runner to
  agents; the tool surface is now 27 tools.
- The Web program panel can build a `.machaon` file from a plain teaching
  sentence list, copy the editor, download a run report, and reopen the run's
  final checkpoint without retyping its path.
- `machaon program --report-out PATH` writes the program report to a chosen
  file instead of `<output>/program-report.json`.
- Program reports now carry `report_format`, `report_format_version` and a
  canonical `report_sha256`. `machaon program-verify` checks the address and
  structure, can require the program file to match the recorded program hash,
  and with `--check-model` reloads the checkpoint and re-runs every fact.
  `--checkpoint` rechecks moved bytes without rewriting the report; `-` reads
  the report from standard input, and `--json` returns integrity failures as
  `{"valid": false, "error": ...}`.
- Python exposes `parse_program_report`, `read_program_report`,
  `write_program_report`, `program_report_sha256` and
  `verify_program_report`; MCP adds `verify_program_report` (28 tools).
- MCP `run_program` accepts the same `out` and `report_out` overrides as the
  CLI runner.
- The Web program panel has an optional report path, displays the report's
  SHA-256, and can open a saved report to verify its address and optionally
  recheck the checkpoint through `/api/program/verify`. It can also match the
  current editor program or recheck the currently selected checkpoint.
- **Build program from facts** now calls `/api/program/new`, the same generator
  and parser used by `machaon program-new`; preserve prompts become `protect`
  entries instead of being silently dropped.
- `machaon program --plan` and `/api/program/plan` now include a side-effect-free
  preflight: model file/directory state, output collision/empty-directory
  status, the report path, and the starter name/size that would be downloaded.
- `examples/tiny_program.py` exercises the complete offline program lifecycle
  on a locally built tiny GPT-2: generate, run, save, reload, verify the report
  and recheck the saved bytes. A skipped-without-extras regression test keeps
  the path honest.
- `machaon.intent`: a deterministic English/Chinese compiler from plain-language
  teaching requests (`->`, labelled blocks, "when asked ...", and sentence
  forms) into an exact, reviewable prompt/answer contract. Unstructured text is
  refused with examples instead of guessed.
- `machaon.recipes.teach_fact`: the one-call teaching loop. It checks whether
  the model already answers, captures an unrelated baseline panel, searches a
  depth window of edit sites, projects each candidate away from the panel,
  applies the best one, and keeps it only when target, budget, panel and KL
  checks pass. Rejected edits are rolled back before return.
- `machaon teach`: one command from a checkpoint and a sentence to a verified
  edited checkpoint, with `--dry-run`, explicit layers, WSP evidence export,
  OpenWSP export and a full JSON report.
- `machaon check`: reload a saved checkpoint and evaluate one prompt/answer pair
  without editing it, with automatic continuation-space handling, JSON output
  and pass/fail exit codes; `--facts` reuses one loaded checkpoint for a whole
  fact list.
- The Web console's **Check a saved fact** panel and `/api/check` expose the
  same saved-checkpoint evaluation without opening or editing a session,
  including a one-fact-per-line list.
- MCP `check_saved` exposes the shared saved-checkpoint check without using the
  model session, including a `facts` list; the tool surface is now 25 tools.
- `HFModel.stored_values` and `Checkpoint.resolve`: a saved checkpoint uses the
  stored orientation and accepts a runtime tensor name whose module prefix
  differs, so GPT-2-style `Conv1D` checkpoints round-trip through teaching.
- Checkpoint writes now copy config and tokenizer files next to the weights and
  remove a partial output when a write fails, so the edited directory loads and
  a failed save leaves no half-checkpoint.
- `machaon intent`: compile and inspect the contract without loading a model.
- `machaon start`: the shortest path to a first edit, with installed-runtime
  diagnostics.
- MCP `parse_intent` and `teach` tools. The server now exposes 24 tools.
- The Web console's **ONE SENTENCE** box, with `/api/teach/preview` and
  `/api/teach`; it previews the parse and runs the same teaching recipe,
  optionally writing the accepted checkpoint.
- Four-language quickstart, CLI, API, MCP, Web and roadmap coverage for the
  intent/teaching layer.
- `machaon.fetch` and `machaon fetch`: a standard-library-only starter
  downloader with Hugging Face / hf-mirror / ModelScope fallback, safetensors
  header validation, resume, a fixed starter catalogue and SSE progress when
  driven by the Web console.
- `machaon.first` and `machaon first`: the zero-to-one path that downloads
  SmolLM2-135M when needed, opens it, teaches one sentence, verifies it and
  saves the accepted checkpoint.
- `machaon.teach_many` and `machaon teach-many`: fact-list teaching with one
  full search per fact, a cumulative budget, original-panel top-1/KL drift
  checks, per-step evidence and a last-globally-safe checkpoint.
- `machaon.checkpointing.save_session`: the shared session-to-checkpoint writer
  used by the CLI, Web console and multi-fact chain.
- The Web **FIRST EDIT** card chains starter download → open session →
  prefilled sentence → teaching result and keeps a prefilled checkpoint output
  plus a Save button after the edit is accepted, and then offers **Open saved
  checkpoint** / **Done**; opening returns to **Teach another fact** with a
  cleared input and the new output path after re-evaluating the taught answer
  against the reopened checkpoint; **Download starter** and **Teach a list**
  expose the same flows, with a live SSE progress bar for downloads. The card
  also chooses between SmolLM2-135M, Qwen2.5-0.5B and GPT-2 when no checkpoint
  is selected, with the multilingual option visible before the first download,
  and its example chips show both an applied edit and an already-known fact.
  Applied edits expose the chosen layer, relative Δ, panel preservation and
  KL p95 directly in the card. Continuation checkpoints use numbered
  `<name>-machaon-N` paths rather than stacking `-machaon` suffixes, and the
  two starter selectors stay in sync. A saved card now offers **Teach a list**
  with the current fact prefilled, and list results expose per-step prompt,
  status, layer, delta and global drift. The console remembers the last
  successfully opened or saved checkpoint across refreshes. The first-run guide
  now introduces the zero-to-one FIRST EDIT card before the advanced loop, and
  the card's ready message names a restored checkpoint.
- `/api/health` reports whether `torch`, `transformers` and `safetensors` are
  importable, and the console shows the exact missing-extra install command.
- `machaon.executor`: one explicit dispatcher for the declarative edit DSL,
  with operand validation, solver refusal, and declared-budget enforcement.
- `machaon.wsp`: WSP acceptance protocol with deterministic same-support,
  same-norm random controls, headroom comparison, and repair-vs-acquisition
  classification.
- `machaon.headroom`: reusable generation and first-token evaluators, including
  a serialisable `DecodingConfig` for sampled / longer-budget baselines.
- `machaon.evidence`: canonical WSP evidence bundles with SHA-256 verification,
  plus `wsp --save` and `wsp-verify` CLI steps.
- `machaon.batch`: least-squares batch edits for many key/value pairs with one
  delta, optional protected-key projection and explicit clamp reporting.
- `machaon.action`: the L1 constrained Python action space. A declared
  `def edit(W): ...` is AST-checked, run in a resource-capped worker process,
  then rejected or admitted by the same shape, budget and protected-key checks.
- `Session.propose_action` and MCP `propose_action` put a validated action on
  the existing pending / apply / rollback path without writing before approval.
- `machaon.mechanism`: loop, empty-output and token-limit metrics, exact
  generation stop traces from `HFModel.generate_records`, and a matched-control
  obstruction gate.
- WSP accepts an optional mechanism payload. When supplied, a target gain is
  rejected unless the configured mechanism metrics improve over the base and
  beat the matched control.
- `Session.assess_generation_wsp` and MCP `assess_generation_wsp` evaluate
  candidate, base, random control and optional headroom on a generation task,
  then attach the resulting mechanism gate and restore the candidate state.
- `machaon.cli wsp-bench` and `machaon.bench.wsp`: labelled
  obstruction-vs-capability benchmark harness with per-case evidence and a
  summary confusion matrix.
- `machaon.planning` and `machaon plan`: evidence-driven edit planning with
  explicit `edit`, `diagnose` and `do_not_edit` routes, plus causal-patching
  site-score ingestion.
- `search_layers` and `Session.search_layers`: live apply/evaluate/rollback
  search over candidate layers, with per-trial evidence and fail-closed cleanup.
- `Session.search_rome_layers` and MCP `search_rome_layers`: an agent-callable
  ROME site search that solves, evaluates and rolls back each candidate layer.
- `diagnose_failure` and `machaon diagnose`: conservative obstruction vs
  capability-gap classification from target and mechanism headroom, plus an
  optional combined diagnosis-and-plan result.
- WSP benchmark cases now carry an explicit `source_kind`; measured cases
  require checkpoint/dataset hashes, timestamp, seed and evaluation script.
  `wsp-bench --require-measured` refuses synthetic fixtures.
- OpenWSP export now supports `form="lowrank"` with float32 factors and sparse
  bf16 correction entries, while preserving existing sparse byte identity.
- `machaon.symmetry`: NumPy-only row/column permutation alignment, positive
  scaling matching, and consistent hidden-unit alignment across an MLP pair.
- `merge_mlp_aligned`: permutation-aware linear interpolation for the two
  matrices around an MLP hidden layer.
- `mlp_outputs` and `functional_alignment_check`: probe-based functional
  verification of hidden-unit alignment.
- `machaon.task_arithmetic`: task-vector extraction and signed coefficient
  combination with explicit relative-norm budget and clamp reporting.
- `machaon.fingerprint`: deterministic tensor/model fingerprints and sparse
  before/after diff reports for audit logs and edit detection.
- A measured threshold-classifier reference case generated by
  `scripts/make_wsp_reference_case.py`, used to test the measured provenance
  gate end to end without claiming LLM evidence.
- The Web console now exposes the batch proposal path, including exact/clamped
  state, target/rank evidence and a collapsible per-key error table.
- The Web console now exposes the WSP audit path with label, reason, control
  gap and headroom gap evidence.
- `wsp_save` / `/api/wsp/save` and a Web **Save evidence** action persist the
  audit report as a content-addressed bundle.
- `/api/wsp/session` and the Web **Assess session** action build the WSP report
  from the current session's candidate, base and random control.
- `/api/wsp/session_evidence` and the Web **Assess session & save evidence**
  action combine the session assessment and content-addressed evidence write.
- `Session.export_openwsp` and `/api/session/export_openwsp` export applied
  deltas in OpenWSP v1 and optionally bind `evidence_sha256` in the manifest.
- `machaon.binding`, `machaon binding` and MCP `verify_binding` verify the edit
  address, payload address and evidence binding as one fail-closed check.
- `/api/binding/verify` and the Web **Verify binding** action expose the same
  edit ↔ evidence check in the browser console.
- `machaon wsp` CLI command and `wsp_audit` MCP tool, both fail-closed on
  missing gates.
- `examples/wsp_report.json`: a zero-download example of a gate-passing
  `repair` decision.

### Changed

- Evaluation docs now distinguish `repair`, `acquisition`, `non_specific`,
  `no_effect`, `regression`, `unclassified` and `inconclusive`.

### Fixed

- The Web **Download starter** control now honors **Open after download**, and
  the FIRST EDIT card continues from its automatic download into open and teach
  instead of stopping at the downloaded path.

## [0.4.3] - 2026-11-08

### Added

- A first-run 60-second guide and a persistent four-step onboarding strip
  explain the inspect → open → baseline → propose → apply → evaluate →
  save/rollback workflow, with a top-bar Guide button for reopening it.
- The guide explains what Inspect, Open session, Capture baseline, Propose,
  Apply, Evaluate, Save and Rollback each do, with a concrete first-edit
  recipe.

### Changed

- The hero copy now states the product's purpose in plain language before
  introducing the contract vocabulary.

## [0.4.2] - 2026-11-08

### Added

- The Web console now ships in English, Simplified Chinese, Japanese and
  Korean, with a top-bar language picker, browser-language detection,
  `localStorage` persistence and `?lang=zh|ja|ko` deep links.
- Static and dynamic console copy share one locale catalog, including gate
  metrics, task statuses, timeline messages and validation errors.
### Changed

- The Web UI uses a CJK-capable system font stack and preserves the
  model-agnostic workflow and checkpoint terminology across all four locales.

## [0.4.1] - 2026-11-08

### Added

- The Web console exposes **Densify from edit layer** for quantized gradient
  solves, matching the Qwen3.6-27B working point.
- The Web console persists applied deltas through `Checkpoint.write_tensor`
  or `Checkpoint.write_with` without mutating the source checkpoint.

### Fixed

- Isolated the Web checkpoint-writer test from optional adapter imports so the
  CI Web job runs in a clean FastAPI environment.

## [0.4.0] - 2026-11-08

### Added

- Local browser console (`machaon-web`) with a three-column workflow:
  checkpoint inspection, executable edit contract, and fail-closed evidence.
  It is a thin FastAPI / SSE front end over the existing `machaon.tools` and
  `machaon.session` functions.
- One-time token authentication, loopback-only binding by default, a
  checkpoint path allowlist, and a single-worker job registry so model loads
  and weight writes cannot race.
- Three-column static UI with checkpoint summaries, bounded tensor listings,
  session state, proposal evidence, gate verdicts, checkpoint persistence and a
  transaction timeline.
- Four-language Web console reference pages and a browser-console quickstart.
- Quantized `Session.open(..., quantize="int4" | "int8")` loading through
  Transformers `QuantoConfig`, with a CPU working point for Qwen3.6-27B.
- Session state now exposes `quantize` and `has_pending`, making the proposal
  phase visible to the browser and to agents.
- `Session.changes()` exposes the current delta map, and the Web console's
  **Persist** action writes it through the checkpoint adapter.
- Web API tests for authentication, path refusal, job lifecycle, SSE, session
  editing and fail-closed gates.

### Changed

- The package version, citation and documentation navigation now describe the
  Web console entry point alongside the CLI and MCP server.

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
