# 零空间投影

这是整个项目建立在一个想法上。

## 问题

把 key `k` 映射到 value `v` 的 rank-one 编辑，是单个约束下的最小范数解：

```
ΔW = (v − W k) kᵀ / (kᵀ k)
```

它对 `k` 精确，对任何其他输入没有任何保证。实践中，它会把邻近 key 移动足以造成损伤的程度。

## 修复

令 `K = [k₁ … kₙ]ᵀ` 是你拒绝改变的 key，`Q` 是 `span(K)` 的一组正交基。定义投影 `P = I − Q Qᵀ`，沿投影后的 key 写编辑：

```
k⊥  = P k
ΔW  = (v − W k) k⊥ᵀ / (k⊥ᵀ k)
```

现在：

- **目标**：`ΔW k = v − W k`，所以 `W' k = v`。精确。
- **保护**：对每个受保护 key，`ΔW kᵢ = 0`。精确到机器精度。

注意分子用的是**原始** `k`。两边都用 `k⊥` 是常见 bug：它能把值打进去，但会多花不必要的范数。

## 实现说明

热路径里从不显式构造 `d_in × d_in` 投影矩阵：

```python
Q = nullspace_basis(preserve)          # (d_in, r)
k_perp = project_complement(k, Q)      # k - Q Qᵀ k，O(d_in · r)
```

当 `d_in = 4096`、`r = 32` 时，基只占 0.5 MB，而不是 128 MB 的稠密矩阵。

## 什么时候会失败

**目标 key 落在受保护子空间里。** 此时 `k⊥ ≈ 0`，没有任何编辑能同时满足两个约束。Machaon 会直接报错，而不是悄悄退化：

```
ValueError: target key lies in the preserved subspace: the edit cannot be made
without changing protected behaviour
```

这是信号，不是 bug：要么缩小受保护集合，要么换一个 key。

## 保证的边界

保证是**一阶的，并且在受保护集合上精确**：对你提供的向量，`ΔW kᵢ = 0` 成立到机器精度。邻近输入只在一阶意义上被保留，剩余部分由事后检查测量：[`verify`](../reference/api.md) 检查 key，KL 检查无关 prompt。

把它看成一把握持宽度已知的手术刀，而不是“其他东西一定没动”的证明。
