# 架构

Machaon 是三层几乎互不认识的系统。

```
                    ┌──────────────────────────┐
   edit spec  ───►  │  machaon.linalg          │  纯 NumPy，无模型
   (DSL, JSON)      │  rank1 · null-space · ...│
                    └───────────┬──────────────┘
                                │ delta (ndarray)
                    ┌───────────▼──────────────┐
                    │  machaon.transaction     │  逐字节回滚
                    │  machaon.store           │  内容寻址
                    └───────────┬──────────────┘
                                │ applied / rejected
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
┌───────▼───────┐     ┌─────────▼─────────┐   ┌─────────▼─────────┐
│ machaon.runtime│     │ machaon.hygieia   │   │ machaon.interop   │
│ HF / safetensors│    │ gates · runner    │   │ OpenWSP v1        │
└───────────────┘     └───────────────────┘   └───────────────────┘
```

## 两个契约

**Runtime。** `next_logits(items) -> (n, vocab)` 和 `nll(items) -> (n,)`。这就是整个面向模型的表面。`HFModel` 在 Hugging Face causal LM 上实现它；测试后端四行就能实现。

**Edit。** `(W, k, v, preserve) -> (W_new, delta)`。纯数组。没有框架类型、没有 device、没有 dtype 意见：内部全部 float64。

契约足够窄，所以加一个架构只需加一个 `LAYER_PATHS` 条目，加一种编辑只需加一个纯函数。

## 数学层为什么是 NumPy

- 任何机器都能测，包括没有 torch 的 CI；
- delta 可以跨框架移植；
- 保证（`ΔW k_i = 0`）是关于数字的陈述，不是关于框架的。

Torch 只出现在 `runtime/` 和 `adapters/` 的边界。

## 为什么用事务而不是原地编辑

回滚是编辑的一部分。`Transaction` 在应用前复制原始字节，因此 `rollback()` 精确还原。`commit(verdict)` 在 verdict 未通过时拒绝提交，使默认行为是失败关闭。

## 验收门为什么失败关闭

缺失指标就是失败。否则通过套件最简单的方法就是停止测量，而任何让测量可选的工具最终都会被在没有测量的情况下使用。

## 格式边界

`machaon.interop.openwsp` 是一个外部格式的自包含实现。它对照参考实现生成的产物做一致性测试，而不是对照描述。如果上游地址变了，测试会失败，而不是悄悄漂移。

## 设计边界

- **闭式优先。** 模型权重从不做梯度下降；ROME 求解只优化一个 value 向量，然后把 delta 交给数学层。
- **文本在边界。** 文本只通过 runtime 的 tokenizer 进入；编辑层只看到数组。
- **编排交给调用方。** `Session` 和 MCP 服务提供状态机——open → baseline → propose → apply → evaluate → rollback——调度留给拥有任务的 agent 或脚本。
