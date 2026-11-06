# 快速开始

五分钟，无需下载。这里的每条命令都能离线运行。

## 1 · 运行 demo

```bash
python -m machaon.cli demo
```

```
op                         rel.delta   target err   preserve drift probe drift p95
----------------------------------------------------------------------------------
rank1 (naive)                 0.0509     1.60e-16         8.71e-02          0.0902
constrained (no cap)          0.0730     2.02e-16         3.11e-16          0.1394
constrained (capped)          0.0200     2.84e-01         8.55e-17          0.0382
```

看第三列。朴素编辑把受保护 key 移动了 8.7e-2；受约束编辑把它们留在 3.8e-16。

## 2 · 编辑一个你拥有的矩阵

```python
import numpy as np
from machaon import rank1_constrained
from machaon.verify import check_protected

rng = np.random.default_rng(0)
W = rng.normal(0, 0.5, size=(64, 48))      # (d_out, d_in)
keys = rng.normal(size=(16, 48))           # 不能移动
k = rng.normal(size=48)                    # 必须记住
v = W @ k + rng.normal(0, 1.0, size=64)    # 必须输出

W_new, delta = rank1_constrained(W, k, v, preserve=keys, max_rel_norm=0.05)

print(check_protected(W, delta, keys))
```

```
{'passed': True, 'max_drift': 3.1e-16, ...}
```

## 3 · 放进事务

```python
from machaon import Workspace

ws = Workspace()
ws.register("layer.8.mlp.c_proj", W)

txn = ws.begin("first edit")
txn.apply("layer.8.mlp.c_proj", delta)

# 任何带布尔 `.passed` 的对象都可以作为 verdict
txn.commit()          # 或 txn.rollback()
```

`commit(verdict)` 在 `verdict.passed` 为 false 时拒绝提交并还原原始字节。

## 4 · 在真实模型上运行

```bash
export HF_ENDPOINT=https://hf-mirror.com
python examples/gpt2_rome.py
```

```
model      : gpt2 (12 layers)
edit       : transformer.h.8.mlp.c_proj.weight
delta      : rel 0.0153
  rewrite_acc  1
  paraphrase_acc 1
  kl_p95       0.000614724
  ppl_ratio    1.00079
PASS (5 gates)
```

## 下一步

- [编辑一个事实](../guides/edit-a-fact.md) —— 完整流程，逐步拆解。
- [评估一次编辑](../guides/evaluate.md) —— 验收门、KL、基准面板。
- [数学原理](../MATH.md) —— 推导、诊断与适用范围。
