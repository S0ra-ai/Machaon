# Machaon 的定位

不同任务需要不同工具。这一页是简图：每种方法改什么、能撤销什么、局部性证据来自哪里，以及 Machaon 在哪里补上契约。

| 方法 | 改动单位 | 撤销 | 局部性证据 | 最适合 |
|---|---|---|---|---|
| **Machaon** | 一个定向、内容寻址的 delta | 逐字节回滚 | 受保护 key 漂移 + KL + 验收门 | 可审计的单次编辑、agent 工作流 |
| 微调 | 全部权重，通过梯度 | 恢复检查点 | 基准套件 | 广泛能力或风格改变 |
| LoRA / PEFT | adapter 权重 | 移除 adapter | 基准套件 | 参数高效适配 |
| ROME / MEMIT | rank-one 或批量 delta | 默认没有 | 论文特定的探针 | 事实编辑研究 |
| 激活 steering | 运行时 hook | 移除 hook | 探针 prompt | 推理时控制 |
| mergekit / task arithmetic | 整模型合并 | 保留两个父模型 | 合并评估 | 组合已训练模型 |
| 剪枝 / 重连 | 结构 | 保留原始检查点 | 基准套件 | 缩小体积或降延迟 |

Machaon 刻意保持窄：它负责**定向编辑**以及围绕它的**验收契约**。其他行仍然只有一次调用之遥，它们的产物可以进入同一套审计：

- mergekit 或 task-arithmetic delta 可以用 `machaon.verify` 检查，用 `machaon.tools.apply_delta` 应用；
- 其他工具生成的 OpenWSP 编辑可以读取、校验哈希并通过验收门；
- LoRA adapter 提供等步数、同层的同类 baseline，见 [R9](RESULTS.md#r9-equal-budget-lora-baseline)；
- steering 方向可以用 `bake_direction` 烘进权重，然后像任何 delta 一样审计。

## Machaon 给参数编辑补上的东西

1. **声明的约束。** 受保护 key 是编辑的一部分，不是事后祈祷。
2. **预算。** `Σ‖ΔWᵢ‖/‖W‖` 在 commit 时记账，超支拒绝。
3. **事务。** 每次编辑都能逐字节撤销。
4. **判决。** 八个失败关闭的验收门，缺失指标算失败。
5. **可移植记录。** OpenWSP v1，逐字节一致，导入时校验哈希。
6. **agent 接口。** 十二个 MCP 工具，文件级和模型级。

## 一句话选择

- 改一个事实或一种行为，并且需要证据：**Machaon**。
- 改整个模型的风格或能力：**微调**。
- 加一个小型可训练 adapter：**LoRA**。
- 组合两个已训练模型：**mergekit**。
- 推理时 steering，不写权重：**激活 steering**。
