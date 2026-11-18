# 评估一次编辑

不要报告一个无法从干净 checkout 复现的数字。

## Runner

`EditSuite` 描述要测什么，runner 负责测。两者都与后端无关。

```python
from machaon.hygieia import EditSuite, evaluate_edit, judge

suite = EditSuite(
    target_items=["The Eiffel Tower is located in the city of"],
    target_token_ids=[tokenizer(" Rome").input_ids[0]],
    paraphrase_items=[...],                   # base 已经回答 " Paris" 的 prompt
    paraphrase_token_ids=[...],
    neutral_items=[...],                      # KL 面板
    corpus_items=[...],                       # 困惑度面板
)

metrics = evaluate_edit(before, after, suite)
verdict = judge(before, after, suite)         # 应用默认验收门
```

## 指标定义

| 指标 | 定义 |
|---|---|
| `rewrite_acc` | 下一个 token 等于期望值的比例 |
| `paraphrase_acc` | 同上，但在留出改写上 |
| `kl_p95` | 中性 prompt 上 `KL(P_before ‖ P_after)` 的 95 分位 |
| `kl_max` | 最差单个中性 prompt |
| `ppl_ratio` | 留出文本上的 `exp(mean NLL_after − mean NLL_before)` |

KL 用分位数报告，不用均值：一个受损 prompt 比一百个未受损 prompt 更重要。

## 基准面板

临时 prompt 告诉你工具是否工作，基准告诉你模型是否还工作。

```bash
python scripts/fetch_benchmarks.py        # 每个文件记录 provenance
python examples/gpt2_bench.py --edits 8
```

| 面板 | 数据集 | 许可证 |
|---|---|---|
| 编辑有效性 | zsRE development | 见[基准测试](../reference/benchmarks.md) |
| 困惑度 | WikiText-2 raw test | CC-BY-SA-3.0 AND GFDL |
| 能力 | MMLU test | MIT |

## 如何读输出

```
== zsRE (edits: 8, layer 8, mean rel.delta 0.0114) ==
  rewrite   0.000 -> 0.500
  rephrase  0.000 -> 0.889   (513 held-out paraphrases)

== WikiText-2 (held out, 20 documents) ==
  perplexity ratio  1.0806

== MMLU (100 questions) ==
  accuracy  0.220 -> 0.220   (delta +0.000)
```

- 有效性是真的：8 个 rewrite 中 4 个翻转，513 个改写中 456 个翻转。
- 代价是真的：困惑度上升 8%。
- GPT-2 上的 MMLU 是粗信号；1.5B+ 面板能分辨更细的差异。报告它，但不要过度解读。

## 修复，还是获得能力？

目标指标上升可能来自三种完全不同的原因：移除了一个障碍、增加了原本不存在的能力，
或者任何同尺寸扰动都会带来类似变化。Machaon 的 WSP 协议把这三者区分开：

```python
from machaon.wsp import assess_wsp

report = assess_wsp(
    base={"rewrite_acc": 0.20},
    candidate={"rewrite_acc": 0.96, ...},          # 完整 gate 指标
    controls={"random_same_support": {"rewrite_acc": 0.20, ...}},
    headroom={"name": "thinking_off", "rewrite_acc": 0.94, ...},
)
assert report["label"] == "repair"
```

`random_controls(deltas)` 会生成与候选编辑同形状、同非零支撑、同 Frobenius 范数的
确定性随机对照。**headroom** 是“未编辑但配置更强”的基线，例如关闭 thinking 的
原始模型。没有 headroom 时，即使提升真实且对照无效，也只会标为 `unclassified`，
不会冒充 `acquisition`。

完整 JSON 协议也可以直接作为 CLI 步骤：

```bash
python -m machaon.cli wsp examples/wsp_report.json
```

| 标签 | 含义 | 是否接受 |
|---|---|---|
| `repair` | 候选达到已有 headroom | gate 通过则接受 |
| `acquisition` | 候选同时超过对照与 headroom | gate 通过则接受 |
| `non_specific` | 同尺寸对照也达到候选 | 不接受 |
| `no_effect` | 增益低于阈值 | 不接受 |
| `regression` | 候选比 base 更差 | 不接受 |
| `unclassified` | 有特异性提升，但没有 headroom | 不接受 |
| `inconclusive` | 缺少对照或必要数据 | 不接受 |

这比“指标涨了”严格得多。报告时应同时给出标签、control gap、headroom gap 与
gate verdict。

### 机制证据

准确率上升时，原始失败机制可能仍然存在。附加 mechanism 报告后，循环率、
token 截顶率和空输出率会进入验收决策：

```python
from machaon import generation_mechanism

evidence = generation_mechanism(
    base_records, candidate_records,
    controls={"random": control_records},
)
report = assess_wsp(
    base=base, candidate=candidate, controls=controls, headroom=headroom,
    mechanism=evidence,
)
```

`generation_metrics` 从文本计算 `loop_rate` 与 `empty_rate`；
`HFModel.generate_records` 进一步提供精确的 `truncation_rate` 和生成 token 数。
`assess_mechanism` 要求候选对每个配置指标都优于 base，并超过最佳匹配对照。
一旦提供 mechanism 证据，机制门失败时 WSP 验收会失败关闭。

要比较障碍修复与能力缺口样本，使用带标签的 benchmark harness：

```bash
python -m machaon.cli wsp-bench examples/wsp_benchmark.json
python -m machaon.cli wsp-bench measured.json --require-measured
```

示例是合成的契约 fixture；在声称模型结论前必须替换为真实测量报告。
measured case 必须提供 checkpoint/dataset 哈希、时间戳、seed 与评测脚本。

### 在 Session 中运行 headroom 基线

当 headroom 需要不同解码路径时，把回调传给 `Session.assess_wsp`。
Machaon 会先恢复未编辑权重，在 base runtime 上调用你的 evaluator，
再在 `finally` 中恢复候选状态：

```python
def evaluate_base(runtime):
    # 示例：关闭 thinking 跑你的 benchmark/tokenizer。
    return {"accuracy": run_benchmark(runtime, thinking=False)}

report = session.assess_wsp(
    prompts, targets, headroom_runner=evaluate_base
)
```

也可以直接使用 `Session.run_headroom(fn)`。这样 headroom 的解码/评测逻辑
留在调用方，而 session 保证权重状态被恢复。

常见场景可以直接用工厂，不必手写回调：

```python
from machaon.headroom import DecodingConfig, make_generation_headroom

headroom = make_generation_headroom(
    prompts, expected_answers,
    DecodingConfig(max_new_tokens=512, do_sample=True,
                   temperature=0.7, top_p=0.9, seed=0),
    match="contains",
    name="sampled_512",
)
report = session.assess_wsp(prompts, targets, headroom_runner=headroom)
```

### 保存判定

WSP 判定可以保存为规范化的 evidence bundle：

```bash
python -m machaon.cli wsp examples/wsp_report.json --save evidence/
python -m machaon.cli wsp-verify evidence/
```

bundle 包含 report、可选 metadata、格式/版本，以及对规范文档计算的 SHA-256
地址。`wsp-verify` 会重算地址；标签或指标被改动会直接拒绝，不会静默接受。

## 长文本连贯性

短答案指标会漏掉最常见的失败：模型仍然答对，但生成文本崩了。

```python
from machaon.hygieia import LONGFORM_GATES, longform_report
from machaon.hygieia.gates import evaluate

report = longform_report(pristine, edited, prompts, max_new_tokens=120)
evaluate(report, LONGFORM_GATES)
```

| 指标 | 定义 |
|---|---|
| `longform_nll_ratio` | 编辑模型文本在**原始模型**下的 NLL，相对 baseline NLL |
| `longform_repetition_delta` | 编辑后 4-gram 重复率减 baseline |
| `longform_empty_rate` | 空生成比例 |

不需要 judge model，不需要额外下载。局部编辑应让 ratio 接近 1.0，重复率 delta 接近 0.0。

GPT-2 上 Eiffel Tower 编辑的实测：

```
baseline_repetition      0.6210
longform_repetition      0.5783
longform_repetition_delta -0.0427
longform_nll_ratio       1.0889
longform_empty_rate      0.0000
PASS (3 gates)
```

## 我们遵守的规则

1. 只编辑 base model **答错**的条目，避免选择偏差。
2. 使用 base model 本来就能正确回答的改写集。
3. 每个数字旁边记录 provenance——数据集、revision、许可证、SHA-256。
4. 报告**没有**改善的部分。
