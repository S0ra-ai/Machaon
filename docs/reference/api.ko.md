# Python API

특별한 언급이 없으면 모두 최상위 패키지에서 가져올 수 있습니다.

## 편집 · `machaon.linalg`

```python
rank1(W, k, v) -> (W_new, delta)
rank1_constrained(W, k, v, preserve=None, max_rel_norm=None) -> (W_new, delta)
bake_direction(W, a_ref, d, alpha) -> (W_new, delta)
ablate_output(W, d) -> (W_new, delta)
ablate_input(W, d) -> (W_new, delta)
low_rank_scale(delta, rank, scale=1.0) -> delta_r
```

규약: `W`는 `(d_out, d_in)`, `k`는 `(d_in,)`, `v`는 `(d_out,)`. 내부는 모두 float64이며 이후 목표 dtype으로 되돌리세요.

| 함수 | 불변량 |
|---|---|
| `rank1` | `W_new @ k == v` |
| `rank1_constrained` | `W_new @ k == v` **그리고** 모든 보호 `kᵢ`에 대해 `delta @ kᵢ == 0` |
| `bake_direction` | `W_new @ a_ref == W @ a_ref + alpha * d_hat` |
| `ablate_output` | `d_hat @ W_new == 0` |
| `ablate_input` | `W_new @ d_hat == 0` |

헬퍼: `nullspace_basis(preserve)`, `project_complement(k, Q)`, `nullspace_projector(preserve)`, `relative_norm(delta, W)`, `clamp_rel_norm(delta, W, max_rel_norm)`.

## 검증 · `machaon.verify`

```python
relative_change(base, delta) -> float
protected_drift(base, delta, keys) -> ndarray     # key별
check_protected(base, delta, keys, max_drift=1e-6) -> dict
audit_edit(deltas, bases, protected=None, max_drift=1e-6) -> dict
```

`check_protected`는 `{passed, max_drift, mean_drift, p95_drift, failed_keys, n_keys, threshold}`를 반환합니다. 다른 도구가 만든 delta에도 작동합니다.

## 트랜잭션 · `machaon.transaction`

```python
ws = Workspace()
ws.register(name, weight)
ws.weight(name); ws.names

txn = ws.begin(label="")
txn.apply(name, delta)
txn.rollback()
txn.commit(verdict=None) -> bool
```

`commit`은 `verdict.passed`가 false일 때 롤백하고 `False`를 반환합니다.

## 예산 · `machaon.budget`

```python
budget = EditBudget(max_total=0.10, max_per_tensor=None)
budget.register(name, weight)
budget.relative(name, delta) -> float
budget.check([(name, delta), ...]) -> BudgetState      # 예외, 상태 변경 없음
budget.add([(name, delta), ...]) -> BudgetState        # 검사 후 기록
budget.state() -> BudgetState
```

`BudgetState`는 `total`, `limit`, `remaining`, `per_tensor`, `tensor_limit`을 가집니다. `BudgetExceeded`는 아무것도 기록하기 전에 발생합니다.

`Workspace(budget=...)`는 각 tensor를 자동 등록하고 `commit()` 시 청구합니다. 거부되면 트랜잭션이 롤백되고 예산은 변하지 않습니다.

## 저장소 · `machaon.store`

```python
store = DeltaStore(root)
sha = store.put(delta, manifest)
store.get(sha); store.manifest(sha); store.list(); sha in store
```

파일은 자신의 바이트 SHA-256으로 이름이 붙습니다. `get`은 기본적으로 내용 주소를 검증하고 손상 시 `StoreCorrupted`를 발생시킵니다.

## 스냅샷 · `machaon.telesphorus`

```python
save(workspace, path); ws = load(path); diff(before, after)
```

## Schema · `machaon.schema`

```python
EditSpec(op, target, intent, rank=1, solver="closed_form", constraints=...)
EditSpec.from_dict(payload).validate()
spec.to_dict()
```

`op`는 `rank1`, `rank1_constrained`, `bake_direction`, `ablate_output`, `ablate_input`, `low_rank_scale` 중 하나. `intent`는 필수이며 이 편집이 존재하는 이유의 감사 기록입니다.

## 런타임 · `machaon.runtime`

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
rt.densify(name) -> bool
rt.densify_layer(layer) -> int
rt.densify_from(layer) -> int
rt.param_count()
```

`items`는 문자열(tokenizer 필요) 또는 token id. `read`/`write`는 항상 `(d_out, d_in)`을 쓰며 `(in, out)`을 저장하는 `transformers.Conv1D`도 처리합니다.

### 지원 모델 패밀리

| 패밀리 | decoder blocks | MLP 투영 | 가중치 레이아웃 |
|---|---|---|---|
| GPT-2 | `transformer.h` | `c_proj` | `Conv1D`의 `(in, out)` |
| GPT-NeoX / Pythia | `gpt_neox.layers` | `mlp.dense_4h_to_h` | `(out, in)` |
| OPT | `model.decoder.layers` | `fc2` | `(out, in)` |
| MPT | `transformer.blocks` | `ffn.down_proj` | `(out, in)` |
| Llama / Mistral / Qwen2 / Qwen3 / Gemma / Phi-3 | `model.layers` | `mlp.down_proj` | `(out, in)` |

CI는 GPT-2, Llama, Mistral, Gemma, Qwen3, Phi-3를 tiny config로 테스트합니다. 실제 체크포인트는 GPT-2 124M, Qwen2.5 0.5B / 1.5B / 3B / 7B(7B는 체크포인트와 평가 모두), Llama-3.1-8B를 포함합니다.

## Procedure · `machaon.procedures`

```python
result = rome_edit(runtime, prompt, target, layer,
                   preserve_prompts=(), steps=250, lr=0.02,
                   module_suffix="mlp.down_proj", max_rel_norm=None)
```

`RomeResult(module, layer, relative_delta, target_probability, target_token_id, steps, diagnostics)`를 반환하고 `runtime`을 인플레이스로 수정합니다.

## 게이트 · `machaon.hygieia`

```python
GATES
evaluate(metrics, gates=None) -> Verdict
Verdict(passed, failures, missing, metrics)

EditSuite(target_items, target_token_ids, neutral_items=(), corpus_items=(),
          paraphrase_items=(), paraphrase_token_ids=(), extra={})
evaluate_edit(before, after, suite) -> dict
judge(before, after, suite, gates=None) -> Verdict

kl_divergence(p, q); kl_summary(p, q); perplexity_ratio(a, b)
softmax(logits); bootstrap_ci(values)
```

## 장문 · `machaon.hygieia.longform`

```python
LONGFORM_GATES
repeated_ngram_rate(text, n=4) -> float
longform_report(pristine, edited, prompts, max_new_tokens=128, n=4) -> dict
```

`longform_repetition`, `baseline_repetition`, `longform_repetition_delta`, `longform_nll_ratio`, `longform_empty_rate`와 샘플을 반환합니다. 일관성은 **편집되지 않은** 모델로 채점합니다.

## 메모리 · `machaon.memory`

```python
available_bytes() -> int | None
weights_bytes(summary) -> int
check_fit(summary, fraction=0.6) -> dict
```

`check_fit`은 가중치가 사용 가능 메모리의 60%를 넘으면 거부합니다. 기본값은 실측에서: 7.6B bf16 체크포인트가 보고된 사용 가능 메모리의 67%에서 스래싱해 프로세스를 멈출 때까지 진행하지 못했습니다. 빨리 실패하는 것이 swap보다 친절합니다.

## Session · `machaon.session`

```python
session = Session.open(path, dtype="float32", budget=EditBudget(0.10))
session.state() -> SessionState(model, parameters, layers, edited, budget_total, ...)
session.baseline(neutral_prompts, corpus_texts=()) -> dict
session.propose(prompt, target, layer, **solve_kwargs) -> dict   # 풀기만, 쓰지 않음
session.apply() -> dict                                          # 쓰고 예산 청구
session.evaluate(prompts, targets) -> dict                       # 정확도, KL, ppl ratio
session.rollback() -> dict                                       # 원본 바이트 복원
session.token_id(text) -> int
```

`propose`는 가중치를 건드리지 않습니다: agent는 커밋 전에 비용(`relative_delta`)과 풀이 확률을 확인할 수 있습니다. `apply`는 예산을 원자적으로 청구하고 초과 시 거부합니다. `rollback`은 편집된 모든 tensor를 복원하고 예산을 재설정합니다.

## 도구 · `machaon.tools`

MCP 서버 뒤의 함수. 모델을 로드하지 않습니다.

```python
describe_checkpoint(path) -> dict
list_tensors(path, contains=None, suffix=None, layer=None, limit=50) -> list
audit_checkpoint_edit(base_path, edit_dir, max_rel_norm=None) -> dict
apply_delta(base_path, tensor, delta_path, out_dir, refuse_rescale=True) -> dict
gate_metrics(metrics) -> dict
```

`audit_checkpoint_edit`는 OpenWSP v1 편집의 두 콘텐츠 주소를 base와 대조하고 상대 변화를 독립적으로 재계산합니다.

## 체크포인트 · `machaon.adapters`

```python
ckpt = Checkpoint(path)
ckpt.names(); ckpt.shards
ckpt.info(name) -> TensorInfo(name, shape, dtype)
ckpt.find(contains=None, suffix=None, layer=None)
ckpt.read(name, dtype=float32); ckpt.read_float64(name)
ckpt.write_with({name: delta}, out_dir) -> report
ckpt.write_tensor(name, values, out_dir, link_others=True) -> report
```

## 거부 규칙 · `machaon.refusals`

```python
check_tensor(name, value, original, frozen=(), refuse_rescale=True)
```

동결 tensor, 모양 불일치, 비유한 값, bfloat16 오버플로, tensor 전체 재스케일에서 `EditRefused`를 발생시킵니다.

## 상호운용 · `machaon.interop.openwsp`

```python
read_edit(path, originals) -> (manifest, tensors)
export_edit(originals, edited, path, base_model=None, revision=None) -> manifest
delta_from_edit(manifest, rebuilt, originals) -> dict
payload_sha256(arrays); edit_sha256(manifest); tensor_sha256(tensor)
bf16_bits(values); bits_float32(bits); is_openwsp_edit(path)
```

## 벤치마크 · `machaon.bench`

```python
load_zsre(path, limit=None) -> [EditCase(case_id, question, answer, rephrases)]
load_mmlu(path, limit=None, subjects=None) -> [MCQItem(question, choices, answer, subject)]
load_wikitext(path, min_chars=400, limit=None) -> [str]
mmlu_accuracy(runtime, items) -> {accuracy, mean_margin, n}

Provenance(name, source, revision, license, sha256, n_items, retrieved, notes)
write_provenance(path, record); read_provenance(path); verify(path)
file_sha256(path)
```
