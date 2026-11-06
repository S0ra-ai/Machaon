# 编辑一个事实

完整流程，与 `machaon.procedures.rome` 的实现一致。

## 六个步骤

### 1 · 选择矩阵

事实通常住在中间层的 MLP 块里。从深度的 60–70% 开始；如果编辑损伤能力，就往更浅的层移动。

```python
from machaon.runtime import HFModel

runtime = HFModel(model, tokenizer)
name = runtime.edit_target(layer=8)          # transformer.h.8.mlp.c_proj.weight
```

`edit_target` 跨模型族翻译名称（`down_proj` ⇄ `c_proj`），并归一化权重方向，因此数学层永远看到 `(d_out, d_in)`。

### 2 · 捕获 key

key 是矩阵**实际**看到的输入，而不是最后一层 layer norm 之前的 residual stream。

```python
ids = runtime.encode(["The Eiffel Tower is located in the city of"])[0]
key = runtime.capture_module_input("transformer.h.8.mlp.c_proj", ids)[0]
```

如果事实不绑定在单一模板上，对多个改写取平均。

### 3 · 构建受保护集合

```python
neutral = ["The capital of Germany is", "Water freezes at a temperature of", ...]
preserve = np.stack([runtime.capture_module_input(module, runtime.encode([p])[0])[0]
                     for p in neutral])
```

这些是不能移动的 key。8 到 30 个通常足够；保护越多，有效性代价越大。

### 4 · 求解 value

value 是矩阵对该 key 应该产生的输出。ROME 用一小段梯度优化它——只针对**单个向量**，从不针对模型：

```python
result = rome_edit(runtime, prompt, target, layer=8,
                   preserve_prompts=neutral, steps=250)
```

### 5 · 带预算应用

```python
W_new, delta = rank1_constrained(W, key, value, preserve=preserve,
                                 max_rel_norm=0.02)
```

预算是对 `‖ΔW‖/‖W‖` 的硬上限。如果编辑需要更多，它会少交付，而不是超支。

### 6 · 验证

```python
from machaon.verify import check_protected
check_protected(W, delta, preserve)     # max_drift 应在 ~1e-16
```

## 端到端运行

```bash
export HF_ENDPOINT=https://hf-mirror.com
python examples/gpt2_rome.py
```

```
model      : gpt2 (12 layers)
edit       : transformer.h.8.mlp.c_proj.weight
solved p(t): 0.9968 after 250 steps
delta      : rel 0.0153
  rewrite_acc  1
  paraphrase_acc 1
  kl_p95       0.000614724
  ppl_ratio    1.00079
PASS (5 gates)
```

## 失效模式

| 症状 | 原因 | 修复 |
|---|---|---|
| 精确 prompt 有效，改写失败 | key 绑定单一模板 | 对多个改写取平均 |
| 能力下降 | 层太居中 | 往浅层移动 2–4 层 |
| bf16 下什么都没变 | delta 被舍入吃掉 | 用 float64 计算 |
| 无关 prompt 漂移 | 没有受保护集合 | 加一个 |
| 报 "preserved subspace" | target key 在受保护张成空间内 | 缩小集合 |
