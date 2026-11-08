# Python API

特に断らない限り、すべてトップレベルパッケージからインポートできます。

## 編集 · `machaon.linalg`

```python
rank1(W, k, v) -> (W_new, delta)
rank1_constrained(W, k, v, preserve=None, max_rel_norm=None) -> (W_new, delta)
bake_direction(W, a_ref, d, alpha) -> (W_new, delta)
ablate_output(W, d) -> (W_new, delta)
ablate_input(W, d) -> (W_new, delta)
low_rank_scale(delta, rank, scale=1.0) -> delta_r
```

規約：`W` は `(d_out, d_in)`、`k` は `(d_in,)`、`v` は `(d_out,)`。内部はすべて float64。後で目標 dtype に戻してください。

| 関数 | 不変量 |
|---|---|
| `rank1` | `W_new @ k == v` |
| `rank1_constrained` | `W_new @ k == v` **かつ** すべての保護 `kᵢ` で `delta @ kᵢ == 0` |
| `bake_direction` | `W_new @ a_ref == W @ a_ref + alpha * d_hat` |
| `ablate_output` | `d_hat @ W_new == 0` |
| `ablate_input` | `W_new @ d_hat == 0` |

ヘルパー：`nullspace_basis(preserve)`、`project_complement(k, Q)`、`nullspace_projector(preserve)`、`relative_norm(delta, W)`、`clamp_rel_norm(delta, W, max_rel_norm)`。

## 契約合成 · `machaon.contract`

```python
from machaon import synthesize_contract

result = synthesize_contract(
    W, k, v, candidates, max_rel_norm=0.02,
    weights=None,                  # 任意の優先度
)
result.delta                       # 正確な更新
result.selected; result.rejected   # 保護済み / 明示的に未保護の key
result.relative_norm; result.target_error
```

貪欲な admission は、正確な更新が予算内に収まる限り候補 key を追加します。返る更新は目標を正確に満たし、`selected` の各 key で正確に不変です。`rejected` は明示されるため、部分保護を完全保護として扱いません。無保護更新でも予算を超える場合は `BudgetExceeded` を送出します。

## 検証 · `machaon.verify`

```python
relative_change(base, delta) -> float
protected_drift(base, delta, keys) -> ndarray     # key ごと
check_protected(base, delta, keys, max_drift=1e-6) -> dict
audit_edit(deltas, bases, protected=None, max_drift=1e-6) -> dict
```

`check_protected` は `{passed, max_drift, mean_drift, p95_drift, failed_keys, n_keys, threshold}` を返します。他のツールが生成した delta にも使えます。

## トランザクション · `machaon.transaction`

```python
ws = Workspace()
ws.register(name, weight)
ws.weight(name); ws.names

txn = ws.begin(label="")
txn.apply(name, delta)
txn.rollback()
txn.commit(verdict=None) -> bool
```

`commit` は `verdict.passed` が false のときロールバックして `False` を返します。

## 予算 · `machaon.budget`

```python
budget = EditBudget(max_total=0.10, max_per_tensor=None)
budget.register(name, weight)
budget.relative(name, delta) -> float
budget.check([(name, delta), ...]) -> BudgetState      # 例外、状態変更なし
budget.add([(name, delta), ...]) -> BudgetState        # 検査して記録
budget.state() -> BudgetState
```

`BudgetState` は `total`、`limit`、`remaining`、`per_tensor`、`tensor_limit` を持ちます。`BudgetExceeded` は何かを記録する前に送出されます。

`Workspace(budget=...)` は各 tensor を自動登録し、`commit()` 時に課金します。拒否時はトランザクションがロールバックされ、予算は変わりません。

## ストレージ · `machaon.store`

```python
store = DeltaStore(root)
sha = store.put(delta, manifest)
store.get(sha); store.manifest(sha); store.list(); sha in store
```

ファイルは自身のバイトの SHA-256 で命名されます。`get` は既定で内容アドレスを検証し、破損時は `StoreCorrupted` を送出します。

## スナップショット · `machaon.telesphorus`

```python
save(workspace, path); ws = load(path); diff(before, after)
```

## Schema · `machaon.schema`

```python
EditSpec(op, target, intent, rank=1, solver="closed_form", constraints=...)
EditSpec.from_dict(payload).validate()
spec.to_dict()
```

`op` は `rank1`、`rank1_constrained`、`bake_direction`、`ablate_output`、`ablate_input`、`low_rank_scale` のいずれか。`intent` は必須で、この編集が存在する理由の監査記録です。

## ランタイム · `machaon.runtime`

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

`items` は文字列（tokenizer が必要）または token id。`read`/`write` は常に `(d_out, d_in)` を使い、`(in, out)` を保存する `transformers.Conv1D` も処理します。

### 対応モデル族

| 族 | decoder blocks | MLP 射影 | 重みレイアウト |
|---|---|---|---|
| GPT-2 | `transformer.h` | `c_proj` | `Conv1D` の `(in, out)` |
| GPT-NeoX / Pythia | `gpt_neox.layers` | `mlp.dense_4h_to_h` | `(out, in)` |
| OPT | `model.decoder.layers` | `fc2` | `(out, in)` |
| MPT | `transformer.blocks` | `ffn.down_proj` | `(out, in)` |
| Llama / Mistral / Qwen2 / Qwen3 / Gemma / Phi-3 | `model.layers` | `mlp.down_proj` | `(out, in)` |

CI は GPT-2、Llama、Mistral、Gemma、Qwen3、Phi-3 を tiny config でテスト。実チェックポイントは GPT-2 124M、Qwen2.5 0.5B / 1.5B / 3B / 7B（7B はチェックポイントと評価の両方）、Llama-3.1-8B をカバーします。

## Procedure · `machaon.procedures`

```python
result = rome_edit(runtime, prompt, target, layer,
                   preserve_prompts=(), steps=250, lr=0.02,
                   module_suffix="mlp.down_proj", max_rel_norm=None)
```

`RomeResult(module, layer, relative_delta, target_probability, target_token_id, steps, diagnostics)` を返し、`runtime` をインプレースで変更します。

## ゲート · `machaon.hygieia`

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

## 長文 · `machaon.hygieia.longform`

```python
LONGFORM_GATES
repeated_ngram_rate(text, n=4) -> float
longform_report(pristine, edited, prompts, max_new_tokens=128, n=4) -> dict
```

`longform_repetition`、`baseline_repetition`、`longform_repetition_delta`、`longform_nll_ratio`、`longform_empty_rate` とサンプルを返します。一貫性は**未編集**モデルで採点します。

## メモリ · `machaon.memory`

```python
available_bytes() -> int | None
weights_bytes(summary) -> int
check_fit(summary, fraction=0.6) -> dict
```

`check_fit` は重みが利用可能メモリの 60% を超えると拒否します。既定値は実測から：7.6B bf16 チェックポイントが報告利用可能メモリの 67% でスラッシングし、プロセスを止めるまで進みませんでした。早く失敗する方が swap より親切です。

## Session · `machaon.session`

```python
session = Session.open(path, dtype="float32", budget=EditBudget(0.10))
session.state() -> SessionState(model, parameters, layers, edited, budget_total, ...)
session.baseline(neutral_prompts, corpus_texts=()) -> dict
session.propose(prompt, target, layer, **solve_kwargs) -> dict   # 解く、書かない
session.apply() -> dict                                          # 書く、予算を課金
session.evaluate(prompts, targets) -> dict                       # 精度、KL、ppl ratio
session.rollback() -> dict                                       # 元バイトを復元
session.token_id(text) -> int
```

`propose` は重みに触れません：agent はコミット前にコスト（`relative_delta`）と求解確率を確認できます。`apply` は予算を原子的に課金し、超過時に拒否します。`rollback` は編集済み tensor を復元し、予算をリセットします。

## ツール · `machaon.tools`

MCP サーバー背後の関数。モデルをロードしません。

```python
describe_checkpoint(path) -> dict
list_tensors(path, contains=None, suffix=None, layer=None, limit=50) -> list
audit_checkpoint_edit(base_path, edit_dir, max_rel_norm=None) -> dict
apply_delta(base_path, tensor, delta_path, out_dir, refuse_rescale=True) -> dict
gate_metrics(metrics) -> dict
```

`audit_checkpoint_edit` は OpenWSP v1 編集の 2 つの内容アドレスを base と照合し、相対変化を独立に再計算します。

## チェックポイント · `machaon.adapters`

```python
ckpt = Checkpoint(path)
ckpt.names(); ckpt.shards
ckpt.info(name) -> TensorInfo(name, shape, dtype)
ckpt.find(contains=None, suffix=None, layer=None)
ckpt.read(name, dtype=float32); ckpt.read_float64(name)
ckpt.write_with({name: delta}, out_dir) -> report
ckpt.write_tensor(name, values, out_dir, link_others=True) -> report
```

## 拒否ルール · `machaon.refusals`

```python
check_tensor(name, value, original, frozen=(), refuse_rescale=True)
```

凍結 tensor、形状不一致、非有限値、bfloat16 オーバーフロー、tensor 全体の再スケールで `EditRefused` を送出します。

## 相互運用 · `machaon.interop.openwsp`

```python
read_edit(path, originals) -> (manifest, tensors)
export_edit(originals, edited, path, base_model=None, revision=None) -> manifest
delta_from_edit(manifest, rebuilt, originals) -> dict
payload_sha256(arrays); edit_sha256(manifest); tensor_sha256(tensor)
bf16_bits(values); bits_float32(bits); is_openwsp_edit(path)
```

## ベンチマーク · `machaon.bench`

```python
load_zsre(path, limit=None) -> [EditCase(case_id, question, answer, rephrases)]
load_mmlu(path, limit=None, subjects=None) -> [MCQItem(question, choices, answer, subject)]
load_wikitext(path, min_chars=400, limit=None) -> [str]
mmlu_accuracy(runtime, items) -> {accuracy, mean_margin, n}

Provenance(name, source, revision, license, sha256, n_items, retrieved, notes)
write_provenance(path, record); read_provenance(path); verify(path)
file_sha256(path)
```
