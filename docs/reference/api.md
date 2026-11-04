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
rt.param_count()
```

`items` may be strings (needs a tokenizer) or token ids. `read`/`write` always
speak `(d_out, d_in)`, including for `transformers.Conv1D` modules that store
`(in, out)`.

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

## Long-form · `machaon.hygieia.longform`

```python
LONGFORM_GATES
repeated_ngram_rate(text, n=4) -> float
longform_report(pristine, edited, prompts, max_new_tokens=128, n=4) -> dict
```

Reports `longform_repetition`, `baseline_repetition`,
`longform_repetition_delta`, `longform_nll_ratio`, `longform_empty_rate`, plus
the samples. Coherence is scored by the **unedited** model: text the base model
finds surprising is evidence of damage.

Repetition is compared as a delta, not absolutely — GPT-2 loops under greedy
decoding before any edit.

## Checkpoints · `machaon.adapters`

```python
ckpt = Checkpoint(path)
ckpt.names(); ckpt.shards
ckpt.info(name) -> TensorInfo(name, shape, dtype)
ckpt.find(contains=None, suffix=None, layer=None)
ckpt.read(name, dtype=float32); ckpt.read_float64(name)
ckpt.write_with({name: delta}, out_dir) -> report
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
export_edit(originals, edited, path, base_model=None, revision=None) -> manifest
delta_from_edit(manifest, rebuilt, originals) -> dict
payload_sha256(arrays); edit_sha256(manifest); tensor_sha256(tensor)
bf16_bits(values); bits_float32(bits); is_openwsp_edit(path)
```

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
