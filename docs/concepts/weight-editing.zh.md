# 权重编辑

一次**编辑**是对一个或多个权重矩阵的加性修改：

```
W' = W + ΔW
```

Machaon 把 `ΔW` 限制在一个**可推导**的族里，使编辑的效果可以被推理，而不是靠猜。

## 动作空间

| 层级 | agent 选择什么 | 数学做什么 | 状态 |
|---|---|---|---|
| L0 | 哪个 procedure、哪一层 | 暂时什么都不做 | ✅ `procedures.rome` |
| L1 | 一个作用于 tensor 的 Python 函数 | 执行它 | 下一步 |
| L2 | 一对 target key/value | rank-one solve | ✅ `rank1_constrained` |
| L3 | 一个可解释方向 | ablation / injection | ✅ `ablate_*`、`bake_direction` |
| L4 | 一组稀疏标量 | 直接赋值 | 通过 `Workspace` |
| L5 | 结构：剪枝、合并、重连 | 由 merge/prune 工具完成；其 delta 可在此审计 | 互操作 |

大多数已发表工作落在 L2 和 L3。L0 把算法变成工具：模型决定**在哪**和**多少**，数学决定**改什么**。

## 一次好编辑的三个属性

### 1. 局部性

改一个 prompt 的答案，不能改变无关 prompt 的答案。这是最先、最响地失效的属性。Machaon 用一个受保护 key 集上的零空间投影来处理它——见[零空间投影](null-space.md)。

用 `‖ΔW kᵢ‖ / ‖W kᵢ‖` 衡量受保护 key，用无关 prompt 上的 KL 散度衡量其余部分。

### 2. 有效性

编辑必须真的做到那件事。在原始 prompt **以及 base model 本来就能同样回答的改写**上衡量。base model 本来答不对的改写集不是泛化测试，是噪声。

### 3. 可逆性

必须能精确撤销。Machaon 在应用前复制原始字节；`rollback()` 逐位还原。

## 一次编辑是什么

- 一次权重修改。模型参数上没有 optimizer：求解是闭式的，或只对单个 value 向量做梯度。
- 持久的。它跨上下文、system prompt 和重启存在。
- 有代价的。每次编辑都花预算：`‖ΔW‖_F / ‖W‖_F`。

## 预算

```
rel(ΔW) = ‖ΔW‖_F / ‖W‖_F
```

经验起点：

| 编辑 | 预算 |
|---|---|
| 事实型，`down_proj` / `c_proj` | 0.01 – 0.03 |
| 行为型，`o_proj` | 0.05 – 0.15 |
| ablation | 不裁剪；改测 KL |

裁剪是刻意的：它打破精确 target 保证，让编辑不超出你授权的模型改动量。
