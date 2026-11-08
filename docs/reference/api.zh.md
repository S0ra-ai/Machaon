# Python API

除非特别说明，所有内容都可以从顶层包导入。

## 编辑 · `machaon.linalg`

```python
rank1(W, k, v) -> (W_new, delta)
rank1_constrained(W, k, v, preserve=None, max_rel_norm=None) -> (W_new, delta)
bake_direction(W, a_ref, d, alpha) -> (W_new, delta)
ablate_output(W, d) -> (W_new, delta)
ablate_input(W, d) -> (W_new, delta)
low_rank_scale(delta, rank, scale=1.0) -> delta_r
```

约定：`W` 是 `(d_out, d_in)`，`k` 是 `(d_in,)`，`v` 是 `(d_out,)`。内部全部用 float64；之后自行转回目标 dtype。

| 函数 | 不变量 |
|---|---|
| `rank1` | `W_new @ k == v` |
| `rank1_constrained` | `W_new @ k == v` **且** 对所有受保护 `kᵢ`，`delta @ kᵢ == 0` |
| `bake_direction` | `W_new @ a_ref == W @ a_ref + alpha * d_hat` |
| `ablate_output` | `d_hat @ W_new == 0` |
| `ablate_input` | `W_new @ d_hat == 0` |

辅助函数：`nullspace_basis(preserve)`、`project_complement(k, Q)`、`nullspace_projector(preserve)`、`relative_norm(delta, W)`、`clamp_rel_norm(delta, W, max_rel_norm)`。

## 契约合成 · `machaon.contract`

```python
from machaon import synthesize_contract

result = synthesize_contract(
    W, k, v, candidates, max_rel_norm=0.02,
    weights=None,                  # 可选优先级
)
result.delta                       # 精确更新
result.selected; result.rejected   # 已保护 / 明确未保护的 key
result.relative_norm; result.target_error
```

贪心准入在精确更新仍满足预算时加入候选 key。返回的更新精确满足目标，并对 `selected` 中每个 key 精确不变；`rejected` 会明确列出，不会把部分覆盖说成完全保护。若连无保护更新都超预算，会抛出 `BudgetExceeded`。

## 验证 · `machaon.verify`

```python
relative_change(base, delta) -> float
protected_drift(base, delta, keys) -> ndarray     # 每个 key
check_protected(base, delta, keys, max_drift=1e-6) -> dict
audit_edit(deltas, bases, protected=None, max_drift=1e-6) -> dict
```

`check_protected` 返回 `{passed, max_drift, mean_drift, p95_drift, failed_keys, n_keys, threshold}`。对任何 delta 有效，包括其他工具生成的编辑。

## 事务 · `machaon.transaction`

```python
ws = Workspace()
ws.register(name, weight)
ws.weight(name); ws.names

txn = ws.begin(label="")
txn.apply(name, delta)
txn.rollback()
txn.commit(verdict=None) -> bool
```

`commit` 在 `verdict.passed` 为 false 时回滚并返回 `False`。

## 预算 · `machaon.budget`

```python
budget = EditBudget(max_total=0.10, max_per_tensor=None)
budget.register(name, weight)
budget.relative(name, delta) -> float
budget.check([(name, delta), ...]) -> BudgetState      # 抛错，不改状态
budget.add([(name, delta), ...]) -> BudgetState        # 先检查再记账
budget.state() -> BudgetState
```

`BudgetState` 携带 `total`、`limit`、`remaining`、`per_tensor`、`tensor_limit`。`BudgetExceeded` 在记录任何内容前抛出。

`Workspace(budget=...)` 自动注册每个 tensor，并在 `commit()` 时记账。被拒时事务回滚，预算不变。

## 存储 · `machaon.store`

```python
store = DeltaStore(root)
sha = store.put(delta, manifest)
store.get(sha); store.manifest(sha); store.list(); sha in store
```

文件以自身字节的 SHA-256 命名。`get` 默认校验内容地址；损坏时抛 `StoreCorrupted`。

## 快照 · `machaon.telesphorus`

```python
save(workspace, path); ws = load(path); diff(before, after)
```

## Schema · `machaon.schema`

```python
EditSpec(op, target, intent, rank=1, solver="closed_form", constraints=...)
EditSpec.from_dict(payload).validate()
spec.to_dict()
```

`op` 是 `rank1`、`rank1_constrained`、`bake_direction`、`ablate_output`、`ablate_input`、`low_rank_scale` 之一。`intent` 必填：它是这次编辑为什么存在的审计记录。

## 运行时 · `machaon.runtime`

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

`items` 可以是字符串（需要 tokenizer）或 token id。`read`/`write` 永远使用 `(d_out, d_in)`，包括存储 `(in, out)` 的 `transformers.Conv1D` 模块。

### 支持的模型族

| 族 | decoder blocks | MLP 投影 | 权重布局 |
|---|---|---|---|
| GPT-2 | `transformer.h` | `c_proj` | `Conv1D` 的 `(in, out)` |
| GPT-NeoX / Pythia | `gpt_neox.layers` | `mlp.dense_4h_to_h` | `(out, in)` |
| OPT | `model.decoder.layers` | `fc2` | `(out, in)` |
| MPT | `transformer.blocks` | `ffn.down_proj` | `(out, in)` |
| Llama / Mistral / Qwen2 / Qwen3 / Gemma / Phi-3 | `model.layers` | `mlp.down_proj` | `(out, in)` |

CI 用 tiny config 覆盖 GPT-2、Llama、Mistral、Gemma、Qwen3 和 Phi-3。真实检查点覆盖 GPT-2 124M、Qwen2.5 0.5B / 1.5B / 3B / 7B（7B 同时覆盖检查点级和评估级）以及 Llama-3.1-8B。

## Procedure · `machaon.procedures`

```python
result = rome_edit(runtime, prompt, target, layer,
                   preserve_prompts=(), steps=250, lr=0.02,
                   module_suffix="mlp.down_proj", max_rel_norm=None)
```

返回 `RomeResult(module, layer, relative_delta, target_probability, target_token_id, steps, diagnostics)`。原地修改 `runtime`。

## 验收门 · `machaon.hygieia`

```python
GATES                          # 默认阈值
evaluate(metrics, gates=None) -> Verdict
Verdict(passed, failures, missing, metrics)

EditSuite(target_items, target_token_ids, neutral_items=(), corpus_items=(),
          paraphrase_items=(), paraphrase_token_ids=(), extra={})
evaluate_edit(before, after, suite) -> dict
judge(before, after, suite, gates=None) -> Verdict

kl_divergence(p, q); kl_summary(p, q); perplexity_ratio(a, b)
softmax(logits); bootstrap_ci(values)
```

## 长文本 · `machaon.hygieia.longform`

```python
LONGFORM_GATES
repeated_ngram_rate(text, n=4) -> float
longform_report(pristine, edited, prompts, max_new_tokens=128, n=4) -> dict
```

报告 `longform_repetition`、`baseline_repetition`、`longform_repetition_delta`、`longform_nll_ratio`、`longform_empty_rate` 和样本。连贯性由**未编辑**模型评分：base model 觉得意外的文本就是变化证据。

## 内存 · `machaon.memory`

```python
available_bytes() -> int | None
weights_bytes(summary) -> int
check_fit(summary, fraction=0.6) -> dict
```

`check_fit` 在权重大于当前可用内存的 60% 时拒绝。默认值来自实测：7.6B bf16 检查点在报告可用内存的 67% 时发生 thrash，直到进程被终止。快速失败比 swap 更友好。

## Session · `machaon.session`

```python
session = Session.open(path, dtype="float32", budget=EditBudget(0.10))
session.state() -> SessionState(model, parameters, layers, edited, budget_total, ...)
session.baseline(neutral_prompts, corpus_texts=()) -> dict
session.propose(prompt, target, layer, **solve_kwargs) -> dict   # 求解，不写入
session.apply() -> dict                                          # 写入并扣预算
session.evaluate(prompts, targets) -> dict                       # 准确率、KL、ppl ratio
session.rollback() -> dict                                       # 还原原始字节
session.token_id(text) -> int
```

`propose` 从不触碰权重：agent 可以在提交前检查代价（`relative_delta`）和求解概率。`apply` 原子扣预算，超预算时拒绝。`rollback` 还原每个已编辑 tensor 并重置预算。

## 工具 · `machaon.tools`

MCP 服务背后的函数。不加载模型。

```python
describe_checkpoint(path) -> dict
list_tensors(path, contains=None, suffix=None, layer=None, limit=50) -> list
audit_checkpoint_edit(base_path, edit_dir, max_rel_norm=None) -> dict
apply_delta(base_path, tensor, delta_path, out_dir, refuse_rescale=True) -> dict
gate_metrics(metrics) -> dict
```

`audit_checkpoint_edit` 对照 base 检查 OpenWSP v1 编辑的两个内容地址，然后独立重算相对变化。

## 检查点 · `machaon.adapters`

```python
ckpt = Checkpoint(path)
ckpt.names(); ckpt.shards
ckpt.info(name) -> TensorInfo(name, shape, dtype)
ckpt.find(contains=None, suffix=None, layer=None)
ckpt.read(name, dtype=float32); ckpt.read_float64(name)
ckpt.write_with({name: delta}, out_dir) -> report
ckpt.write_tensor(name, values, out_dir, link_others=True) -> report
```

## 拒绝规则 · `machaon.refusals`

```python
check_tensor(name, value, original, frozen=(), refuse_rescale=True)
```

在冻结 tensor、形状不匹配、非有限值、bfloat16 溢出或整 tensor 缩放时抛 `EditRefused`。

## 互操作 · `machaon.interop.openwsp`

```python
read_edit(path, originals) -> (manifest, tensors)
export_edit(originals, edited, path, base_model=None, revision=None) -> manifest
delta_from_edit(manifest, rebuilt, originals) -> dict
payload_sha256(arrays); edit_sha256(manifest); tensor_sha256(tensor)
bf16_bits(values); bits_float32(bits); is_openwsp_edit(path)
```

## 基准 · `machaon.bench`

```python
load_zsre(path, limit=None) -> [EditCase(case_id, question, answer, rephrases)]
load_mmlu(path, limit=None, subjects=None) -> [MCQItem(question, choices, answer, subject)]
load_wikitext(path, min_chars=400, limit=None) -> [str]
mmlu_accuracy(runtime, items) -> {accuracy, mean_margin, n}

Provenance(name, source, revision, license, sha256, n_items, retrieved, notes)
write_provenance(path, record); read_provenance(path); verify(path)
file_sha256(path)
```
