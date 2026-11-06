# 事务与验收门

## 事务

一次无法撤销的编辑不是编辑，是一次部署。

```python
ws = Workspace()
ws.register("layer.8.mlp.c_proj", W)

txn = ws.begin("eiffel tower -> rome")
txn.apply("layer.8.mlp.c_proj", delta)
txn.commit(verdict)
```

- `apply` 在写入前复制原始数组。同一事务里对同一 tensor 的两次编辑按逆序撤销。
- `rollback()` 逐字节还原每个原始值。
- `commit(verdict)` 在 `verdict.passed` 为 false 时回滚。传 `None` 会无条件提交——只在你没有测量并接受风险时这样做。

已关闭的事务拒绝后续操作；不能提交两次，也不能在 rollback 后继续 apply。

## 验收门

验收门决定一次编辑是否有资格被提交。

| gate | 阈值 | 保护什么 |
|---|---|---|
| `rewrite_acc` | ≥ 0.95 | 编辑确实生效 |
| `paraphrase_acc` | ≥ 0.60 | 泛化到精确 prompt 之外 |
| `unrelated_acc` | ≥ 1.00 | 其他答案没变 |
| `kl_p95` | ≤ 0.05 | 无关 prompt 上的分布漂移 |
| `kl_max` | ≤ 0.50 | 没有单个灾难性离群点 |
| `mmlu_delta` | ≥ −0.01 | 能力保持 |
| `ppl_ratio` | ≤ 1.02 | 没有全局损伤 |
| `longform_contradictions` | = 0 | 长文本生成保持连贯 |

## 失败关闭

```python
from machaon.hygieia import evaluate

verdict = evaluate(metrics)
verdict.passed      # 任何指标缺失时为 False
verdict.missing     # ['mmlu_delta', ...]
```

缺失指标就是失败。否则通过验收最简单的方法就是停止测量，而任何让测量可选的工具，最终都会被在没有测量的情况下使用。

窄实验可以显式传入门子集——但要说明：

```python
evaluate(metrics, {"rewrite_acc": (">=", 0.99), "ppl_ratio": ("<=", 1.05)})
```

## 累计预算

一次编辑便宜，二十次不便宜。Machaon 按 tensor 和全模型跟踪 `Σ‖ΔWᵢ‖ / ‖W‖`，并拒绝超预算的提交。

```python
from machaon import Workspace
from machaon.budget import EditBudget

budget = EditBudget(max_total=0.10)
ws = Workspace(budget=budget)
ws.register("layer.20.mlp.down_proj", W)

txn = ws.begin("edit 10")
txn.apply("layer.20.mlp.down_proj", delta)
txn.commit()          # 抛 BudgetExceeded，回滚，不记账
```

由三角不等式，这个和是 `‖ΣΔWᵢ‖ / ‖W‖` 的上界，因此是保守的：可以放心执行，绝不美化。

在 GPT-2 124M 上实测，每次 rank-one 编辑 rel.Δ ≈ 0.011：

| 编辑数 | 累计 Δ | WikiText-2 困惑度 |
|---|---|---|
| 0 | 0.000 | 1.000 |
| 8 | 0.089 | 1.049 |
| 16 | 0.151 | 1.098 |
| 32 | 0.255 | 1.191 |

预算 `0.10` 会在第 10 次编辑时拒绝。预算在 **commit** 时记账，所以 `rollback()` 和失败 verdict 都不花钱。

## 读懂 verdict

```
FAIL
  missing: longform_contradictions
  kl_p95 = 0.113 violates <= 0.05
```

两个不同的问题。`missing` 表示你没测；另一行表示你测了但没过。按这个顺序修。
