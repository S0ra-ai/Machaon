# 相关工作

这个想法处在五个社区的交汇处。它们各自覆盖了一部分循环，Machaon 把编辑、事务和验收接成一个可操作的整体。

## 参数级知识编辑

| 工作 | 说明 |
|---|---|
| [ROME](https://arxiv.org/abs/2202.05262) | rank-one 事实编辑；这条线的起点 |
| [MEMIT](https://arxiv.org/abs/2210.07229) | 批量编辑数千事实 |
| [MEND](https://arxiv.org/abs/2110.11309) | 预测编辑的超网络 |
| [SERAC](https://arxiv.org/abs/2206.06520) | 用外部记忆代替权重 |
| [AlphaEdit](https://arxiv.org/abs/2410.02355) | 对保留知识做零空间投影；Machaon 沿用该投影，并在外层提供事务与验收契约 |
| [Survey](https://arxiv.org/abs/2310.19704) | 进入文献的最佳入口 |
| [EasyEdit](https://github.com/zjunlp/EasyEdit) | 最接近的现有框架（2.9k★），面向批处理脚本 |

## 免训练权重手术

| 工作 | 说明 |
|---|---|
| [Model Surgery](https://arxiv.org/abs/2407.08770) | 极少量参数即可改变行为 |
| [SCALPEL](https://arxiv.org/abs/2601.07411) | 用于能力消融的低秩参数编辑 |
| [ThinkEdit](https://arxiv.org/abs/2503.22048) | 权重编辑改变推理长度 |
| [Resisting Harmful Instructions through Training-Free Weight Editing](https://arxiv.org/abs/2610.21189) | 说明这项能力既实用又具有双用途属性 |
| [TARS](https://arxiv.org/abs/2412.10257) | 定向角度反转，用于知识移除 |

## 表示与激活编辑

| 工作 | 说明 |
|---|---|
| [Refusal is mediated by a single direction](https://arxiv.org/abs/2406.11717) | “一个方向，一种行为”的结果 |
| [RepE](https://arxiv.org/abs/2310.01405) | 表示工程 |
| [CAA](https://arxiv.org/abs/2312.06681) | 对比激活加法 |
| [ActAdd](https://arxiv.org/abs/2306.03341) | 激活工程 |
| [SAELens](https://github.com/decoderesearch/SAELens) | 把稀疏自编码器特征作为编辑目标 |

## 权重空间代数

| 工作 | 说明 |
|---|---|
| [Task Arithmetic](https://arxiv.org/abs/2212.04089) | 通过任务向量编辑模型 |
| TIES / DARE | 干扰感知的合并 |
| [mergekit](https://github.com/arcee-ai/mergekit) | 成熟工具链（7.4k★） |
| [Evolutionary model merge](https://github.com/SakanaAI/evolutionary-model-merge) | 搜索合并配方 |

## LLM 驱动的优化

| 工作 | 说明 |
|---|---|
| [OPRO](https://arxiv.org/abs/2309.03409) | 把 LLM 当优化器 |
| [TextGrad](https://arxiv.org/abs/2406.07496) | 文本梯度 |
| [SEAL](https://arxiv.org/abs/2506.10943) | 模型自己生成更新指令 |
| [Foundation models understand weights](https://arxiv.org/abs/2503.00838) | 支持“LLM 能推理权重空间” |
| [LLM as a Tool, Not an Agent](https://arxiv.org/abs/2604.16555) | 对 agentic editing 最尖锐的反驳 |
| [Agentic NAS](https://arxiv.org/abs/2607.07984) | 对立阵营 |

## 构建前值得读的两篇

1. [Revisiting Parameter-Based Knowledge Editing: Theoretical Limits and Empirical Evidence](https://arxiv.org/abs/2606.00570) —— 论证参数编辑存在根本上限。
2. [Long-form evaluation of model editing](https://arxiv.org/abs/2402.09394) —— 通过短 QA 的编辑会在长生成中崩塌。

这两篇是 `machaon.hygieia` 验收门设计的承重墙。

## 空白

1. 现有工作多是 Python API 或脚本，缺少 agent 可操作的动作空间。
2. 缺少事务层——没有东西负责快照、验证和回滚。
3. 评估不是一等公民。
4. 缺少让 LLM **规划手术**的系统：诊断 → 选位点 → 声明约束 → 交给闭式求解器。
