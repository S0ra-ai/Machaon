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
