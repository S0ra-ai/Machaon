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

## 对称性对齐 · `machaon.symmetry`

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

`align_rows` 按余弦相似度把候选 unit 匹配到参考 unit，返回置换、对齐后的张量、
逐 unit 相似度与 `passed`。匹配采用置信度优先的贪心，不是 Hungarian 精确解；
结果会记录 `optimal=False`。正缩放会被归一化消除。`align_mlp_hidden` 会把同一个
hidden-unit 置换同时应用到输入行和输出列。旋转、attention-head 对称性与跨架构
对齐不在本模块范围内。
`merge_mlp_aligned` 会在做线性插值前应用该置换；`alpha=0` 返回参考模型，
`alpha=1` 返回对齐后的候选模型。
`functional_alignment_check` 在探针输入上评估两个 MLP，报告绝对/相对输出误差；
超过 `atol` 时 verdict 失败关闭。

## Task arithmetic · `machaon.task_arithmetic`

```python
TaskVector(delta, relative_norm, base_norm)
task_vector(base, finetuned) -> TaskVector
combine_task_vectors(vectors, coefficients=None, base=None,
                     max_rel_norm=None, clamp=True)
    -> TaskVectorCombination
```

`task_vector` 计算显式差值及其相对幅度。`combine_task_vectors` 支持正负系数，
输入可以是 `TaskVector` 或原始 delta。提供 `base` 时会报告组合后的相对范数；
超过 `max_rel_norm` 时 `clamp=True` 会缩放并标记，`clamp=False` 会抛错。
这是算术层，不是合并配方，也不是评测结论。

## 权重指纹 · `machaon.fingerprint`

```python
TensorFingerprint(sha256, shape, size, norm, mean, std, zero_fraction)
tensor_fingerprint(value) -> TensorFingerprint
model_fingerprint(weights) -> dict
diff_tensors(base, changed) -> dict
```

SHA-256 地址覆盖规范化 float64 shape header 与 tensor 字节。
`model_fingerprint` 对排序后的 tensor 名称/地址做聚合，因此 mapping 顺序不影响
结果。`diff_tensors` 报告变化条目数、support 比例、相对变化、最大绝对变化、
变化行列和前后地址。它用于检测内容变化，不建立真实性或签名。

## 可执行编辑 DSL · `machaon.executor`

`EditSpec` 不只是元数据。`execute_spec` 把已验证的 spec 分派到同一组纯 NumPy
原语，并在返回前检查声明的相对范数预算：

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
result.to_dict()          # JSON 安全摘要；不包含数组
```

`solver="search"` 会被拒绝；`closed_form` 与 `donor` 可执行。executor 不写
checkpoint，也不训练。

## 受约束动作 · `machaon.action`

`ActionSpec` 接收一个声明式单张量编辑函数：

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
    intent="移除选定方向",
    max_rel_norm=0.02,
).validate()

result = execute_action(spec, W, keys=k, values=v, preserve=protected)
result.delta; result.relative_norm; result.source_sha256
result.protected; result.to_dict()
```

函数可以使用 `W`、可选的 `keys`/`values`/`preserve`、声明的
`max_rel_norm`、少量纯数组函数、局部赋值和数组算术。导入、循环、类、lambda、
推导式、私有属性和任意调用会在执行前被拒绝。worker 在独立进程中运行，有墙钟
期限与资源上限；父进程会拒绝错误形状、非有限值、超预算 delta 和受保护 key
漂移。`Session.propose_action(...)` 会把 delta 放进与其他编辑相同的
pending / apply / rollback 路径。

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

## 批量编辑 · `machaon.batch`

```python
from machaon import solve_batch

solution = solve_batch(
    W, keys, values,               # keys (n, d_in), values (n, d_out)
    preserve=None,                 # 可选保护 key
    max_rel_norm=0.05,
    clamp=True,                    # False：超预算直接拒绝
    ridge=0.0,
)
solution.delta
solution.relative_norm; solution.target_error
solution.per_key_error; solution.exact; solution.clamped; solution.rank
```

最小 Frobenius 范数最小二乘 delta 会在 key 矩阵行满秩时满足所有目标行。
受保护集合会把目标 key 投影到其零空间，因此 `delta @ preserve.T == 0` 精确成立；
如果投影后目标行降秩，`exact=False` 并报告逐 key 残差，而不是隐藏失败。
`clamp=True` 会把超预算的精确解转成显式非精确解；`clamp=False` 直接抛错。

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

## WSP 验收 · `machaon.wsp`

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

`random_control` 生成与候选 delta 同形状、同非零支撑、同 Frobenius 范数的确定性
随机矩阵。`classify_effect` 输出 `repair`、`acquisition`、`non_specific`、
`no_effect`、`regression`、`unclassified` 或 `inconclusive`。`assess_wsp`
再加入 fail-closed 的 gate verdict：只有 `repair`/`acquisition` 且候选 gate 通过，
`accepted` 才为 true。`assess_report` 解析 CLI 与 MCP 工具使用的 JSON 结构。
提供 `mechanism` 时，其障碍门也会成为接受条件。

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

每个 case 标记为 `obstruction` 或 `capability_gap`。runner 输出逐例 WSP 决策、
是否匹配 `expected_repairable`，以及混淆矩阵摘要。该工具不会把 fixture 冒充为
科学结论，只负责把比较过程变得可复现。`source_kind="measured"` 必须提供
`model`、`checkpoint_sha256`、`dataset`、`dataset_sha256`、`evaluated_at`、
`seed` 与 `evaluation_script`；两个 hash 必须是 64 位小写 SHA-256。
`scripts/make_wsp_reference_case.py` 生成 measured 阈值分类器参考用例，仅用于
验证 pipeline，不是 LLM 能力结论。

## 规划 · `machaon.planning`

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

路线是 `edit`、`diagnose` 或 `do_not_edit`。`capability_gap` 默认走
`do_not_edit`；没有正位点分数的 `obstruction` 走 `diagnose`；只有存在位点证据时
才产出具体 layer 与预算。
`search_layers` 对回调结果排序并保留失败试验。`cleanup` 在每次试验后运行；
若它抛错，整个搜索中止，不会从脏状态继续。

`diagnose_failure` 只有在目标 headroom 与至少一个机制指标同时改善时才判为
obstruction。目标差距小且没有机制改善时判为 capability gap；证据缺失或矛盾时
保持 unknown。

## 机制证据 · `machaon.mechanism`

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

`generation_metrics` 输出重复循环率与空输出率；有 trace 元数据时还输出截顶率
和生成 token 数。`HFModel.generate_records` 提供这些元数据。
`assess_mechanism` 要求每个配置指标都优于 base，并超过最佳匹配对照；
缺少对照或指标时返回 `inconclusive`，不会当作通过。

## Headroom 基线 · `machaon.headroom`

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

工厂返回的回调可直接传给 `Session.assess_wsp(..., headroom_runner=...)`。
`generation_accuracy` 使用 `runtime.generate`；`next_token_accuracy` 与
`Session.evaluate` 的首 token 指标一致。`DecodingConfig` 可序列化，因此
采样 / 更长预算 / 重复惩罚的 headroom 可以随证据报告一起保存。
`include_mechanism=True` 时，报告还会携带循环与截顶证据。

## Evidence bundle · `machaon.evidence`

```python
FORMAT; FORMAT_VERSION
EvidenceError
evidence_sha256(document) -> str
write_evidence(path, report, metadata=None) -> dict   # 文件或目录
read_evidence(path) -> dict                           # 校验地址
is_wsp_evidence(path) -> bool
```

目录目标写入 `evidence.json`；`.json` 目标直接写入。文档是规范 JSON，包含
`format`、`format_version`、`report`、`metadata` 与 `evidence_sha256`。
`read_evidence` 会拒绝被改动的标签、指标或 metadata，而不只是拒绝坏 JSON。

## 绑定校验 · `machaon.binding`

```python
from machaon.binding import verify_binding

report = verify_binding("out/edit", "evidence/")
report.passed; report.edit_sha256; report.evidence_sha256; report.errors
```

`verify_binding` 校验 OpenWSP edit 地址、payload 地址，以及 manifest 中的
`machaon.evidence_sha256` 是否与 evidence bundle 自身地址一致。它不加载
base checkpoint；tensor 级强审计仍使用 `openwsp.read_edit`。

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
session = Session.open(path, dtype="float32", budget=EditBudget(0.10),
                       quantize=None)  # int4 / int8 通过 QuantoConfig
session.state() -> SessionState(model, parameters, layers, edited, budget_total, ...)
session.baseline(neutral_prompts, corpus_texts=()) -> dict
session.propose(prompt, target, layer, **solve_kwargs) -> dict   # 求解，不写入
session.propose_batch(prompts, values, layer, **batch_kwargs) -> dict
session.propose_action(source, intent, layer, **action_kwargs) -> dict
session.apply() -> dict                                          # 写入并扣预算
session.evaluate(prompts, targets) -> dict                       # 准确率、KL、ppl ratio
session.evaluate_wsp(prompts, targets, seed=0) -> dict           # base、candidate、random control
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
session.rollback() -> dict                                       # 还原原始字节
session.changes() -> dict                                        # 每个 tensor 的当前 delta
session.token_id(text) -> int
```

`propose` 从不触碰权重：agent 可以在提交前检查代价（`relative_delta`）和求解概率。`apply` 原子扣预算，超预算时拒绝。`rollback` 还原每个已编辑 tensor 并重置预算。
`assess_generation_wsp` 一次生成 candidate、base、随机对照与可选 headroom，
把停止轨迹送入机制门，并在 `finally` 中恢复候选状态。

## 意图与教学 · `machaon.intent`、`machaon.recipes`

```python
parse_intent(text) -> TeachIntent
teach(session, sentence_or_prompt, answer=None, ...) -> TeachResult
teach_fact(session, prompt, answer, ..., dry_run=False) -> TeachResult
candidate_layers(n_layers, count=7, low=0.55, high=0.95) -> list[int]
TeachPolicy(...) -> policy
```

`parse_intent` 是确定性的、无依赖的教学语法编译器，不连接模型，也不猜测层：

```python
from machaon import parse_intent

intent = parse_intent("教会模型：北京的首都是 → 北京")
intent.prompt, intent.answer       # ('北京的首都是', '北京')
intent.confidence                  # 'high'
intent.to_command("./models/qwen") # 精确的 CLI 调用
```

`teach_fact` 是一次调用完成的配方：要求 session 干净，先检查模型是否已经会回答，
自动捕获无关基线面板，通过 `Session.search_rome_layers` 在深度窗口内逐层尝试，
应用最佳候选，并且只在所有必选本地检查通过时保留编辑。

策略默认值为 `max_rel_norm=0.02`、`max_kl_p95=0.05`、`max_ppl_ratio=1.02`
和 `min_panel_agreement=0.875`。`TeachResult.to_dict()` 保留每次试验、排序、
检查、警告和未测量指标，可直接序列化为 JSON。

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
export_edit(originals, edited, path, base_model=None, revision=None,
            machaon_metadata=None, form="sparse", rank=None) -> manifest
delta_from_edit(manifest, rebuilt, originals) -> dict
payload_sha256(arrays); edit_sha256(manifest); tensor_sha256(tensor)
bf16_bits(values); bits_float32(bits); is_openwsp_edit(path)
```

`form="lowrank"` 写入 float32 因子与稀疏 bf16 修正项。读取端重建精确的 bf16
编辑位，不是近似读取。

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
