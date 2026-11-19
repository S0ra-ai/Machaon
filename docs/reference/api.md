# Python API

Everything is importable from the top-level package unless noted.

## Edits · `machaon.linalg`

```python
rank1(W, k, v) -> (W_new, delta)
rank1_constrained(W, k, v, preserve=None, max_rel_norm=None) -> (W_new, delta)
bake_direction(W, a_ref, d, alpha) -> (W_new, delta)
ablate_output(W, d) -> (W_new, delta)
ablate_input(W, d) -> (W_new, delta)
low_rank_scale(delta, rank, scale=1.0) -> delta_r
```

Conventions: `W` is `(d_out, d_in)`, `k` is `(d_in,)`, `v` is `(d_out,)`.
All computation is float64; cast after.

| function | invariant |
|---|---|
| `rank1` | `W_new @ k == v` |
| `rank1_constrained` | `W_new @ k == v` **and** `delta @ kᵢ == 0` for all protected `kᵢ` |
| `bake_direction` | `W_new @ a_ref == W @ a_ref + alpha * d_hat` |
| `ablate_output` | `d_hat @ W_new == 0` |
| `ablate_input` | `W_new @ d_hat == 0` |

Helpers: `nullspace_basis(preserve)`, `project_complement(k, Q)`,
`nullspace_projector(preserve)`, `relative_norm(delta, W)`,
`clamp_rel_norm(delta, W, max_rel_norm)`.

```python
from machaon import rank1_constrained, nullspace_basis, project_complement
```

## Symmetry alignment · `machaon.symmetry`

```python
permute_rows(matrix, permutation) -> ndarray
permute_columns(matrix, permutation) -> ndarray

align_rows(reference, candidate, axis=0, min_similarity=0.0,
           max_units=4096) -> UnitAlignment
align_mlp_hidden(reference_in, reference_out,
                 candidate_in, candidate_out, ...) -> MLPAlignment
merge_mlp_aligned(reference_in, reference_out,
                  candidate_in, candidate_out, alpha=0.5, ...) -> MergedMLP
mlp_outputs(incoming, outgoing, inputs, activation="relu") -> ndarray
functional_alignment_check(reference_in, reference_out,
                           candidate_in, candidate_out, probes,
                           activation="relu", atol=1e-9) -> dict
```

`align_rows` matches candidate units to reference units by cosine similarity
and returns the permutation, aligned tensor, per-unit similarities and a
`passed` flag. Matching is confidence-first greedy, not a Hungarian solve; the
result records `optimal=False`. Positive scaling is removed by normalisation.
`align_mlp_hidden` applies the same hidden-unit permutation to incoming rows and
outgoing columns. Rotations, attention-head symmetries and cross-architecture
alignment are outside this module.
`merge_mlp_aligned` applies that permutation before linear interpolation;
`alpha=0` returns the reference and `alpha=1` returns the aligned candidate.
`functional_alignment_check` evaluates both MLPs on probe inputs and reports
absolute/relative output error; the verdict fails closed above `atol`.

## Task arithmetic · `machaon.task_arithmetic`

```python
TaskVector(delta, relative_norm, base_norm)
task_vector(base, finetuned) -> TaskVector
combine_task_vectors(vectors, coefficients=None, base=None,
                     max_rel_norm=None, clamp=True)
    -> TaskVectorCombination
```

`task_vector` computes the explicit difference and its relative magnitude.
`combine_task_vectors` accepts positive or negative coefficients and either
`TaskVector` objects or raw deltas. With `base`, the result reports its relative
norm; exceeding `max_rel_norm` clamps with `clamped=True` or raises when
`clamp=False`. This is arithmetic, not a merge recipe or an evaluation result.

## Fingerprints · `machaon.fingerprint`

```python
TensorFingerprint(sha256, shape, size, norm, mean, std, zero_fraction)
tensor_fingerprint(value) -> TensorFingerprint
model_fingerprint(weights) -> dict
diff_tensors(base, changed) -> dict
```

The SHA-256 address is over a canonical float64 shape header plus tensor bytes.
`model_fingerprint` aggregates sorted tensor names/addresses, so mapping order
does not matter. `diff_tensors` reports changed entries, support fraction,
relative change, max absolute change, changed rows/columns and before/after
addresses. This detects content changes; it does not establish authenticity.

## Executable edit DSL · `machaon.executor`

`EditSpec` is not just metadata. `execute_spec` dispatches a validated spec to
the same pure NumPy primitives, then checks the declared relative-norm budget:

```python
from machaon import EditSpec, Target, Constraints, execute_spec

spec = EditSpec(
    op="rank1_constrained",
    target=Target("model.layers.14.mlp.down_proj", 14),
    intent="make France's capital Lyon",
    constraints=Constraints(max_rel_norm=0.02),
).validate()

result = execute_spec(spec, W, k=k, v=v, preserve=keys)
result.delta; result.relative_norm; result.target_error
result.to_dict()          # JSON-safe summary; the array is omitted
```

`solver="search"` is refused; `closed_form` and `donor` are executable. The
executor never writes a checkpoint and never trains.

## Constrained actions · `machaon.action`

`ActionSpec` accepts a declared edit function for one tensor:

```python
from machaon import ActionSpec, Target, execute_action

source = """def edit(W):
    updated, delta = rank1_constrained(
        W, keys, values, preserve=preserve, max_rel_norm=max_rel_norm)
    return delta
"""

spec = ActionSpec(
    source,
    Target("model.layers.14.mlp.down_proj.weight", 14),
    intent="remove the selected direction",
    max_rel_norm=0.02,
).validate()

result = execute_action(spec, W, keys=k, values=v, preserve=protected)
result.delta; result.relative_norm; result.source_sha256
result.protected; result.to_dict()
```

The function may use `W`, optional `keys`, `values` and `preserve`, the declared
`max_rel_norm`, a small set of pure array functions, local assignments and
array arithmetic. Imports, loops, classes, lambdas, comprehensions, private
attributes and arbitrary calls are rejected before execution. The worker runs
in a separate process with a wall-clock deadline and resource caps; the parent
rejects wrong shapes, non-finite values, over-budget deltas and protected-key
drift. `Session.propose_action(...)` puts the delta on the same pending /
apply / rollback path as other edits.

## Contract synthesis · `machaon.contract`

```python
from machaon import synthesize_contract

result = synthesize_contract(
    W, k, v, candidates, max_rel_norm=0.02,
    weights=None,                  # optional priorities
)
result.delta                       # exact update
result.selected; result.rejected   # admitted / reported-unprotected keys
result.relative_norm; result.target_error
```

The greedy admission rule adds candidate keys while the exact update stays
inside the budget. Every returned update satisfies the target exactly and is
exactly invariant on every key in `selected`. Rejected keys are never silently
described as protected. If even the unprotected update exceeds the budget,
`BudgetExceeded` is raised.

## Batch edits · `machaon.batch`

```python
from machaon import solve_batch

solution = solve_batch(
    W, keys, values,               # keys (n, d_in), values (n, d_out)
    preserve=None,                 # optional protected keys
    max_rel_norm=0.05,
    clamp=True,                    # False: refuse an over-budget solve
    ridge=0.0,
)
solution.delta
solution.relative_norm; solution.target_error
solution.per_key_error; solution.exact; solution.clamped; solution.rank
```

The minimum-Frobenius-norm least-squares delta satisfies all target rows when
the key matrix is full row rank. A protected set projects the target keys into
its null space, so `delta @ preserve.T == 0` exactly; if the target rows lose
rank in that projection, `exact=False` and the per-key residual is reported
rather than hidden. `clamp=True` converts an over-budget exact solve into an
explicit inexact solve; `clamp=False` raises.

## Verification · `machaon.verify`

```python
relative_change(base, delta) -> float
protected_drift(base, delta, keys) -> ndarray     # per key
check_protected(base, delta, keys, max_drift=1e-6) -> dict
audit_edit(deltas, bases, protected=None, max_drift=1e-6) -> dict
```

`check_protected` returns `{passed, max_drift, mean_drift, p95_drift,
failed_keys, n_keys, threshold}`.

Works on any delta, including edits produced by other tools.

## Transactions · `machaon.transaction`

```python
ws = Workspace()
ws.register(name, weight)
ws.weight(name); ws.names

txn = ws.begin(label="")
txn.apply(name, delta)
txn.rollback()
txn.commit(verdict=None) -> bool
```

`commit` rolls back when `verdict.passed` is false and returns `False`.

## Budget · `machaon.budget`

```python
budget = EditBudget(max_total=0.10, max_per_tensor=None)
budget.register(name, weight)
budget.relative(name, delta) -> float
budget.check([(name, delta), ...]) -> BudgetState      # raises, changes nothing
budget.add([(name, delta), ...]) -> BudgetState        # checks, then records
budget.state() -> BudgetState
```

`BudgetState` carries `total`, `limit`, `remaining`, `per_tensor`,
`tensor_limit`. `BudgetExceeded` is raised before anything is recorded.

`Workspace(budget=...)` registers every tensor automatically and charges at
`commit()`. On refusal the transaction is rolled back and the budget is
untouched.

## Storage · `machaon.store`

```python
store = DeltaStore(root)
sha = store.put(delta, manifest)
store.get(sha); store.manifest(sha); store.list(); sha in store
```

Files are named by the SHA-256 of their own bytes.

## Snapshots · `machaon.telesphorus`

```python
save(workspace, path); ws = load(path); diff(before, after)
```

## Schema · `machaon.schema`

```python
EditSpec(op, target, intent, rank=1, solver="closed_form", constraints=...)
EditSpec.from_dict(payload).validate()
spec.to_dict()
```

`op` is one of `rank1`, `rank1_constrained`, `bake_direction`, `ablate_output`,
`ablate_input`, `low_rank_scale`. `intent` is mandatory: it is the audit record.

## Runtime · `machaon.runtime`

```python
model = load_pretrained(path, **kwargs)  # text or multimodal AutoModel
rt = HFModel(model, tokenizer=None, device=None)
rt.layer_path; rt.n_layers; rt.layers
rt.edit_target(layer, suffix="mlp.down_proj") -> "…weight"
rt.weight_shape(name) -> (d_out, d_in)
rt.read(name) -> float64 ndarray
rt.write(name, array)
rt.encode(texts) -> (ids, mask)
rt.next_logits(items); rt.next_logits_ids(ids, mask=None)
rt.nll(items); rt.nll_ids(ids, mask=None)
rt.score_continuations([(prompt, continuation), ...]) -> ndarray
rt.capture_module_input(name, ids, position=-1) -> ndarray
rt.densify(name) -> bool
rt.densify_layer(layer) -> int
rt.densify_from(layer) -> int
rt.param_count()
```

`items` may be strings (needs a tokenizer) or token ids. `read`/`write` always
speak `(d_out, d_in)`, including for `transformers.Conv1D` modules that store
`(in, out)`.

### Supported families

| family | decoder blocks | MLP projection | weight layout |
|---|---|---|---|
| GPT-2 | `transformer.h` | `c_proj` | `(in, out)` via `Conv1D` |
| GPT-NeoX / Pythia | `gpt_neox.layers` | `mlp.dense_4h_to_h` | `(out, in)` |
| OPT | `model.decoder.layers` | `fc2` | `(out, in)` |
| MPT | `transformer.blocks` | `ffn.down_proj` | `(out, in)` |
| Llama / Mistral / Qwen2 / Qwen3 / Gemma / Phi-3 | `model.layers` | `mlp.down_proj` | `(out, in)` |
| Qwen3.5 / Qwen3.6 multimodal | `model.language_model.layers` | `mlp.down_proj` | `(out, in)` |

Tested in CI with tiny configs for GPT-2, Llama, Mistral, Gemma, Qwen3 and
Phi-3. Real checkpoints exercised: GPT-2 124M, Qwen2.5 0.5B / 1.5B / 3B / 7B
(7B at both checkpoint and evaluation level) and Llama-3.1-8B.

## Procedures · `machaon.procedures`

```python
result = rome_edit(runtime, prompt, target, layer,
                   preserve_prompts=(), steps=250, lr=0.02,
                   module_suffix="mlp.down_proj", max_rel_norm=None)
```

Returns `RomeResult(module, layer, relative_delta, target_probability,
target_token_id, steps, diagnostics)`. Mutates `runtime` in place.

## Gates · `machaon.hygieia`

```python
GATES                          # default thresholds
evaluate(metrics, gates=None) -> Verdict
Verdict(passed, failures, missing, metrics)

EditSuite(target_items, target_token_ids, neutral_items=(), corpus_items=(),
          paraphrase_items=(), paraphrase_token_ids=(), extra={})
evaluate_edit(before, after, suite) -> dict
judge(before, after, suite, gates=None) -> Verdict

kl_divergence(p, q); kl_summary(p, q); perplexity_ratio(a, b)
softmax(logits); bootstrap_ci(values)
```

## WSP acceptance · `machaon.wsp`

```python
WSPProtocol(target_metric="rewrite_acc", min_gain=0.05,
            control_margin=0.05, headroom_margin=0.05,
            require_gates=True, require_controls=True)

random_control(delta, seed=0, preserve_support=True) -> ndarray
random_controls(deltas, seed=0, preserve_support=True) -> dict

classify_effect(base, candidate, controls=None, headroom=None,
                protocol=None) -> EffectVerdict
assess_wsp(base, candidate, controls=None, headroom=None,
           gates=None, protocol=None, mechanism=None) -> dict
assess_report(payload) -> dict
```

`random_control` returns a deterministic random matrix with the same shape,
nonzero support and Frobenius norm as the candidate delta. `classify_effect`
labels the result `repair`, `acquisition`, `non_specific`, `no_effect`,
`regression`, `unclassified` or `inconclusive`. `assess_wsp` adds the
fail-closed gate verdict; `accepted` is true only for `repair`/`acquisition`
with a passing candidate gate. `assess_report` parses the JSON shape used by
the CLI and MCP tool. When `mechanism` is supplied, its obstruction gate is
also required for acceptance.

## WSP benchmark · `machaon.bench.wsp`

```python
WSPBenchmarkCase(case_id, failure_type, expected_repairable,
                 base, candidate, controls={}, headroom=None,
                 mechanism=None, gates=None, protocol=None,
                 source_kind="synthetic", provenance={}, metadata={})

evaluate_case(case) -> dict
run_benchmark(cases) -> dict
load_benchmark(path) -> list[WSPBenchmarkCase]
```

Cases are labelled `obstruction` or `capability_gap`. The runner reports each
case's WSP decision, whether it matched `expected_repairable`, and a summary
confusion matrix. The helper does not claim that a fixture is a scientific
result; it makes the comparison reproducible. `source_kind="measured"` requires
`model`, `checkpoint_sha256`, `dataset`, `dataset_sha256`, `evaluated_at`,
`seed` and `evaluation_script`; both hashes must be lowercase SHA-256.
`scripts/make_wsp_reference_case.py` generates a measured threshold-classifier
reference for pipeline validation, not an LLM capability claim.

## Planning · `machaon.planning`

```python
EditPlan(route, failure_type, reason, layer=None, module_suffix=None,
         max_rel_norm=None, ranked_layers=(), next_probe="", metadata={})

scores_from_patch_report(patch_report, key="recovery") -> dict
plan_surgery(failure_type, layer_scores=None, patch_report=None,
             module_suffix="mlp.down_proj", max_rel_norm=0.02,
             min_score=0.0, metadata=None) -> EditPlan
search_layers(layers, evaluate, metric="accuracy",
              higher_is_better=True, max_evaluations=None,
              cleanup=None) -> dict
DiagnosisProtocol(target_metric="accuracy", min_target_gap=0.05,
                  min_mechanism_gap=0.05,
                  mechanism_metrics=("loop_rate", "truncation_rate"))
diagnose_failure(base, headroom=None, mechanism=None,
                 protocol=None) -> FailureDiagnosis
diagnose_and_plan(base, headroom=None, mechanism=None,
                  protocol=None, plan_kwargs=None) -> dict
```

Routes are `edit`, `diagnose` and `do_not_edit`. A `capability_gap` always takes
`do_not_edit`; an `obstruction` without a positive site score takes `diagnose`;
only an obstruction with site evidence produces a concrete layer and budget.
`search_layers` ranks callback results and retains failed trials. `cleanup`
runs after every trial; if it raises, the search aborts rather than continuing
from a dirty state.

`diagnose_failure` calls an obstruction only when target headroom and at least
one mechanism metric improve together. A small target gap without mechanism
relief is a capability gap; missing or contradictory evidence remains unknown.

## Mechanism evidence · `machaon.mechanism`

```python
MechanismProtocol(metrics=("loop_rate", "truncation_rate"),
                  directions=(("loop_rate", "lower"),
                              ("truncation_rate", "lower")),
                  min_improvement=0.05, control_margin=0.05,
                  require_controls=True)

generation_metrics(texts, hit_token_limit=None,
                   generated_tokens=None, ngram=4) -> dict
records_metrics(records, ngram=4) -> dict
generation_mechanism(base_records, candidate_records,
                     controls=None, headroom=None) -> dict
assess_mechanism(base, candidate, controls=None,
                 headroom=None, protocol=None) -> MechanismVerdict
```

`generation_metrics` reports repeated-loop rate and empty-output rate, plus
truncation rate and token counts when trace metadata is available.
`HFModel.generate_records` provides that metadata. `assess_mechanism` requires
every configured metric to improve over the base and beat the best matched
control; missing controls or metrics are `inconclusive`, not a pass.

## Headroom baselines · `machaon.headroom`

```python
DecodingConfig(max_new_tokens=128, do_sample=False, temperature=1.0,
               top_p=1.0, top_k=0, repetition_penalty=1.0, seed=None)

generation_accuracy(runtime, prompts, expected, config=None,
                    match="exact", include_samples=False,
                    include_mechanism=False) -> dict
next_token_accuracy(runtime, prompts, targets) -> dict

make_generation_headroom(prompts, expected, config=None,
                         match="exact", name="generation",
                         include_mechanism=False)
make_next_token_headroom(prompts, targets, name="next_token")
```

The factories return callables suitable for
`Session.assess_wsp(..., headroom_runner=...)`. `generation_accuracy` uses
`runtime.generate`; `next_token_accuracy` matches `Session.evaluate`'s
first-token metric. `DecodingConfig` is serialisable, so a sampled / longer /
repetition-penalised headroom can travel with the evidence report. With
`include_mechanism=True`, the report also carries loop and truncation evidence.

## Evidence bundles · `machaon.evidence`

```python
FORMAT; FORMAT_VERSION
EvidenceError
evidence_sha256(document) -> str
write_evidence(path, report, metadata=None) -> dict   # file or directory
read_evidence(path) -> dict                           # verifies the address
is_wsp_evidence(path) -> bool
```

A directory target receives `evidence.json`; a `.json` target is written
directly. The document is canonical JSON with `format`, `format_version`,
`report`, `metadata` and `evidence_sha256`. `read_evidence` rejects a changed
label, metric or metadata field, not just malformed JSON.

## Binding verification · `machaon.binding`

```python
from machaon.binding import verify_binding

report = verify_binding("out/edit", "evidence/")
report.passed; report.edit_sha256; report.evidence_sha256; report.errors
```

`verify_binding` checks the OpenWSP edit address, payload address and the
manifest's `machaon.evidence_sha256` against the evidence bundle's own address.
It does not load the base checkpoint; use `openwsp.read_edit` for the stronger
tensor-level audit.

## Long-form · `machaon.hygieia.longform`

```python
LONGFORM_GATES
repeated_ngram_rate(text, n=4) -> float
longform_report(pristine, edited, prompts, max_new_tokens=128, n=4) -> dict
```

Reports `longform_repetition`, `baseline_repetition`,
`longform_repetition_delta`, `longform_nll_ratio`, `longform_empty_rate`, plus
the samples. Coherence is scored by the **unedited** model: text the base model
  finds surprising is evidence of change.

Repetition is compared as a delta, not absolutely — GPT-2 loops under greedy
decoding before any edit.

## Memory · `machaon.memory`

```python
available_bytes() -> int | None
weights_bytes(summary) -> int
check_fit(summary, fraction=0.6) -> dict
```

`check_fit` refuses a checkpoint whose weights exceed 60% of currently available
memory. The default comes from measurement: a 7.6B bf16 checkpoint at 67% of
reported-available memory thrashed until the run was killed. Failing fast is
kinder than swapping.

## Session · `machaon.session`

```python
session = Session.open(path, dtype="float32", device=None,
                       budget=EditBudget(0.10),
                       quantize=None)  # int4 / int8 via QuantoConfig
session.state() -> SessionState(model, parameters, layers, device,
                                edited, budget_total, ...)
session.baseline(neutral_prompts, corpus_texts=()) -> dict
session.propose(prompt, target, layer, **solve_kwargs) -> dict   # solves, does not write
session.propose_batch(prompts, values, layer, **batch_kwargs) -> dict
session.propose_action(source, intent, layer, **action_kwargs) -> dict
session.apply() -> dict                                          # writes, charges the budget
session.evaluate(prompts, targets) -> dict                       # accuracy, KL, ppl ratio
session.evaluate_wsp(prompts, targets, seed=0) -> dict           # base, candidate, random control
session.run_headroom(evaluate_fn, name="headroom") -> dict
session.assess_wsp(prompts, targets, headroom=None,
                   headroom_runner=None, seed=0) -> dict
session.assess_generation_wsp(prompts, expected, config=None,
                              headroom_config=None, seed=0) -> dict
session.search_layers(layers, proposal_factory, evaluation,
                      metric="accuracy") -> dict
session.search_rome_layers(prompt, target, layers, solver="direction",
                           max_rel_norm=None, metric="accuracy") -> dict
session.export_openwsp(path, base_model=None, revision=None,
                       metadata=None, form="sparse", rank=None) -> manifest
session.rollback() -> dict                                       # restores original bytes
session.changes() -> dict                                        # current deltas by tensor
session.token_id(text) -> int
```

`propose` never touches the weights: an agent can inspect the cost
(`relative_delta`) and the solved probability before committing. `apply`
charges the budget atomically and refuses when it would be exceeded.
`rollback` restores every edited tensor and resets the budget.
`assess_generation_wsp` generates candidate, base, random control and optional
headroom in one pass, feeds their stop traces to the mechanism gate, and
restores the candidate state in a `finally` block.
`quantize` accepts `int4` or `int8` with `dtype="auto"`; it is the working
point for a 27B checkpoint on a laptop-sized memory envelope.
`device` defaults to CPU; use `cuda`, `mps`, or `auto` to opt into a compute
device. `SessionState.device` reports the actual resolved device after load.
`Session.open` first checks that the directory is a complete checkpoint
(`config.json`, weights and tokenizer). An interrupted download is refused with
"rerun machaon fetch to resume it" instead of a tokenizer traceback; the same
check covers the CLI, Web console and MCP server.
`machaon.devices.device_report()` (also exported as `machaon.device_report`)
answers "can this machine use CUDA or MPS?" without loading a model:

```python
from machaon import device_report

device_report()
# {'default': 'cpu', 'torch': True, 'cuda': False, 'cuda_count': 0,
#  'mps': True, 'mps_built': True}
```

## Intent and teaching · `machaon.intent`, `machaon.recipes`

```python
parse_intent(text) -> TeachIntent
teach(session, sentence_or_prompt, answer=None, ...) -> TeachResult
teach_fact(session, prompt, answer, ..., dry_run=False) -> TeachResult
candidate_layers(n_layers, count=7, low=0.55, high=0.95) -> list[int]
TeachPolicy(...) -> policy
```

`parse_intent` is a deterministic, dependency-free compiler for the supported
teaching grammar. It never contacts a model and never guesses a layer:

```python
from machaon import parse_intent

intent = parse_intent("教会模型：北京的首都是 → 北京")
intent.prompt, intent.answer       # ('北京的首都是', '北京')
intent.confidence                  # 'high'
intent.to_command("./models/qwen") # exact CLI invocation
```

`teach` is the sentence-level alias: pass one sentence, or a prompt plus an
explicit answer. `teach_fact` is the one-call recipe. It requires a clean session, checks
whether the model already answers, captures an unrelated baseline panel, tries
a depth window of layers through `Session.search_rome_layers`, applies the best
candidate, and keeps it only when every required local check passes:

```python
from machaon import teach_fact

result = teach_fact(session, "The Eiffel Tower is in the city of", " Rome")
if result.status != "applied":
    print(result.summary, result.next_steps)
```

The policy defaults are `max_rel_norm=0.02`, `max_kl_p95=0.05`,
`max_ppl_ratio=1.02` and `min_panel_agreement=0.875`. `TeachResult.to_dict()`
retains every trial, ranking, check, warning and unmeasured metric. The result
is JSON-safe and is the same object returned by the CLI, MCP and Web entry
points.

```python
from machaon.teach_many import load_facts, teach_many

facts = load_facts("facts.txt")
result = teach_many(
    session_factory, "./models/qwen", facts, "./models/qwen-many",
    budget=0.10, policy=TeachPolicy(max_rel_norm=0.02),
)
result.final, result.cumulative_relative_delta, result.global_panel_agreement
```

`teach_many` runs the full recipe for each fact on a fresh session loaded from
the previous accepted checkpoint. It enforces the cumulative budget, checks
top-1/KL against the original panel, keeps the last globally safe checkpoint,
and writes a per-step report. `save_session(session, out_dir)` is the shared
writer used by the CLI, Web and chain; it applies the runtime's stored
orientation and removes a partial directory on failure.

```python
from machaon.first import first_edit

result = first_edit(
    "The Eiffel Tower is in the city of -> Rome",
    "./models/SmolLM2-135M",          # downloaded if absent
    "./models/SmolLM2-135M-machaon",
)
result.status, result.final, result.model_fetched
```

`first_edit` is the zero-to-one orchestration: fetch the starter when the model
is absent, open it, run `teach_fact`, and save the accepted edit.

```python
from machaon import load_program, run_program

program = load_program("hello.machaon")       # parse without a model
result = run_program(program)                 # teach, save, reload, check
result.status, result.verification["passed"], result.report_path
```

`load_program` accepts the line-oriented text format or JSON and resolves
relative paths from the program file. `run_program` delegates edits to
`teach_many`, writes `<output>/program-report.json`, and by default reloads the
written checkpoint with `check_facts`. Pass `verify=False` to skip only that
last reload; the report still records the program hash and every teaching step.

The saved report has a canonical content address and a matching reader:

```python
from machaon import read_program_report, verify_program_report

document = read_program_report("program-report.json")   # raises on mismatch
checked = verify_program_report(
    "program-report.json",
    program_path="hello.machaon",                       # optional hash cross-check
    check_model=True,                                   # optional live fact check
)
checked["report_sha256"], checked["model_check"]["passed"]
```

`program_report_sha256(document)` computes the address excluded from its own
hash field. `parse_program_report(text)` is the in-memory form used by the Web
API. `verify_program_report(..., checkpoint=...)` can recheck moved bytes
without rewriting the report. An integrity mismatch raises
`ProgramReportError`; a model check that disagrees with the report is returned
as `model_check.matches_report == False` so the caller can decide how to act.

`render_program_text(...)` is the generator behind `machaon program-new`. It
returns the text form of a validated program, including the optional advanced
fields, without writing a file:

```python
from machaon import render_program_text

text = render_program_text(
    model="./models/SmolLM2-135M",
    output="./models/lesson",
    fetch="smol135m",
    teach=["The Eiffel Tower is in the city of -> Rome"],
)
```

`program_preflight(program)` performs the same side-effect-free readiness check
as `--plan`: model file/directory state, output collision and empty-directory
state, `report_path`, and the starter name/size when a fetch is declared.

## Tools · `machaon.tools`

The functions behind the MCP server. No model is loaded.

```python
describe_checkpoint(path) -> dict
list_tensors(path, contains=None, suffix=None, layer=None, limit=50) -> list
audit_checkpoint_edit(base_path, edit_dir, max_rel_norm=None) -> dict
apply_delta(base_path, tensor, delta_path, out_dir, refuse_rescale=True) -> dict
gate_metrics(metrics) -> dict
parse_intent(text) -> dict
```

`audit_checkpoint_edit` verifies both content addresses of an OpenWSP v1 edit
against the base checkpoint, then recomputes the relative change independently.

## Checkpoints · `machaon.adapters`

```python
ckpt = Checkpoint(path)
ckpt.names(); ckpt.shards
ckpt.info(name) -> TensorInfo(name, shape, dtype)
ckpt.find(contains=None, suffix=None, layer=None)
ckpt.read(name, dtype=float32); ckpt.read_float64(name)
ckpt.write_with({name: delta}, out_dir) -> report
ckpt.write_tensor(name, values, out_dir, link_others=True) -> report
```

## Refusals · `machaon.refusals`

```python
check_tensor(name, value, original, frozen=(), refuse_rescale=True)
```

Raises `EditRefused` on frozen tensors, shape mismatch, non-finite values,
bfloat16 overflow, or whole-tensor rescaling.

## Interop · `machaon.interop.openwsp`

```python
read_edit(path, originals) -> (manifest, tensors)
export_edit(originals, edited, path, base_model=None, revision=None,
            machaon_metadata=None, form="sparse", rank=None) -> manifest
delta_from_edit(manifest, rebuilt, originals) -> dict
payload_sha256(arrays); edit_sha256(manifest); tensor_sha256(tensor)
bf16_bits(values); bits_float32(bits); is_openwsp_edit(path)
```

`form="lowrank"` writes float32 factors plus sparse bf16 correction entries.
The reader reconstructs the exact edited bf16 bits, not an approximation.

## Benchmarks · `machaon.bench`

```python
load_zsre(path, limit=None) -> [EditCase(case_id, question, answer, rephrases)]
load_mmlu(path, limit=None, subjects=None) -> [MCQItem(question, choices, answer, subject)]
load_wikitext(path, min_chars=400, limit=None) -> [str]
mmlu_accuracy(runtime, items) -> {accuracy, mean_margin, n}

Provenance(name, source, revision, license, sha256, n_items, retrieved, notes)
write_provenance(path, record); read_provenance(path); verify(path)
file_sha256(path)
```
