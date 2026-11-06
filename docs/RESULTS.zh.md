# 实测结果

每个运行都可以从干净 clone 用一条命令复现，数据集带有记录的哈希。下面是完整记录。

## R1 · GPT-2 124M · 移动一个事实

```bash
HF_ENDPOINT=https://hf-mirror.com python examples/gpt2_rome.py
```

| | |
|---|---|
| 模型 | `gpt2`（124.4M 参数，12 层） |
| 编辑 | `transformer.h.8.mlp.c_proj.weight` |
| 方法 | rank-one + 8 个受保护 key 的零空间投影 |
| value 求解 | 250 步 Adam，只优化一个 **768 维 value 向量**（不碰模型权重） |
| 相对 delta | **0.0153** |
| 求解后 p(target) | 0.9968 |

| 指标 | 值 | gate | |
|---|---|---|---|
| `rewrite_acc` | 1.00 | ≥ 0.99 | pass |
| `paraphrase_acc` | **1.00**（6/6） | ≥ 0.50 | pass |
| `kl_p95` | 0.00061 | ≤ 0.50 | pass |
| `kl_max` | 0.00086 | ≤ 0.50 | pass |
| `ppl_ratio` | 1.0008 | ≤ 1.05 | pass |

六个改写都是 base model **本来**就回答 " Paris" 的 prompt；否则这个数字测不到东西。

| prompt | 之前 | 之后 |
|---|---|---|
| The Eiffel Tower is located in the city of | Paris | **Rome** |
| The Eiffel Tower is in the city of | Paris | **Rome** |
| The Eiffel Tower is situated in the city of | Paris | **Rome** |
| The Eiffel Tower stands in the city of | Paris | **Rome** |
| The Eiffel Tower, located in the city of | Paris | **Rome** |
| The Eiffel Tower is a famous landmark in the city of | Paris | **Rome** |
| Tourists visiting the Eiffel Tower stay in the city of | Paris | **Rome** |
| The capital of Germany is | the | the |
| The capital of Italy is | Rome | Rome |
| Water freezes at a temperature of | about | about |

## R2 · 两个模型，公开基准

同一协议、同样 8 次编辑、同一评估器。Target injection：从数据集中选一个实体 token，把模型编辑成输出它，然后测原问题和人工改写的问法。

| 模型 | 层位 | 平均 rel.Δ | rewrite | rephrase（513） | WikiText-2 | MMLU |
|---|---|---|---|---|---|---|
| GPT-2 124M | 8 / 12 | 0.0135 | 0.500 | 0.259 | 1.066 | 0.220 → 0.220（n=100） |
| Qwen2.5-0.5B | 20 / 24 | 0.0133 | 0.375 | 0.275 | 1.030 | 0.440 → 0.440（n=50） |

### 两个工程发现

| 发现 | 测量 |
|---|---|
| macOS CPU 上的 bf16 backward | 4.66 秒/步 |
| 同模型 float32 backward | **0.18 秒/步** |
| Qwen 第 12 层，一次编辑 | rel. Δ 0.150 |
| Qwen 第 20 层，同一次编辑 | rel. Δ **0.0043** |

## R3 · Qwen2.5-1.5B

与 R2 同协议，100 步求解，float32。

| 指标 | 值 |
|---|---|
| 参数 | 1,543.7M（28 层） |
| 平均相对 delta | **0.0065** |
| rewrite | 0.500 |
| rephrase | 0.273（513） |
| WikiText-2 困惑度 | **1.005** |
| MMLU（n=50） | 0.400 → 0.400 |

文件级写入：338 个 tensor、2.9 GB、2.3 秒，只改一个 tensor，其余逐字节不变（`examples/file_edit.py`）。

## R4 · Qwen2.5-3B

同协议，80 步求解，float32，分片检查点（2 个文件）。

| 指标 | 值 |
|---|---|
| 参数 | 3,085.9M（36 层） |
| 平均相对 delta | **0.0052** |
| rewrite | 0.625 |
| rephrase | 0.230（457） |
| WikiText-2 困惑度 | **1.002** |
| MMLU（n=30） | 0.500 → 0.500 |

单模型副本峰值 RSS 11.7 GB。单模型 benchmark 路径让它在 32 GB 机器上跑得下。

## R5 · 1.5B 长文本

同一编辑，五个 prompt 上 80 token greedy 生成，由未编辑模型评分。

| 指标 | 值 |
|---|---|
| baseline 4-gram 重复率 | 0.097 |
| 编辑后 4-gram 重复率 | 0.196 |
| 重复率 delta | **+0.099**（gate ≤ 0.10） |
| NLL ratio | 0.938 |
| 空生成 | 0.0 |

三项 gate 全过：重复率 delta +0.099 对 0.10 阈值，NLL ratio 0.938，无空生成。

## R6 · Qwen2.5-7B，int8 评估

| 指标 | 值 |
|---|---|
| 参数 | 7,615.6M（28 层） |
| 加载 | 通过 `optimum-quanto` 使用 int8，`lm_head` 保持 bf16 |
| 编辑层 | 26 / 28，与第 27 层一起稠密化 |
| 平均相对 delta | **0.0043** |
| rewrite | **1.000**（3/3） |
| rephrase | **0.929**（39/42） |
| WikiText-2 困惑度 | **1.007** |
| MMLU（n=8） | 0.375 → 0.375 |
| 峰值 RSS | 10.2 GB |

局部窗口让梯度路径可负担：稠密化编辑层及其后续层，其余保持 int8。rel.Δ 从 0.050 降到 0.0043，困惑度从 1.260 降到 1.007。

## R7 · 3B solver sweep

| 设置 | 后续 block 数 | rel.Δ | rewrite | rephrase | WikiText-2 |
|---|---|---|---|---|---|
| 第 26 层，lr 0.02，80 步 | 10 | 0.0052 | 0.625 | 0.230 | 1.002 |
| 第 26 层，lr 0.05，30 步 | 10 | 0.0079 | 0.500 | 0.236 | 1.002 |
| 第 32 层，lr 0.05，30 步 | 4 | 0.0068 | 0.625 | 0.312 | 1.009 |
| 第 34 层，lr 0.05，30 步 | 2 | 0.0072 | 0.625 | 0.327 | 1.009 |

深度帮助泛化：编辑越靠近输出，rephrase 从 0.230 升到 0.327。

**同目标数的 3B/7B 对比修正了容量结论。** 使用与 7B 完全相同的设置（3 个目标、每个 15 个改写、30 步、lr 0.05）：

| 模型 | rel.Δ | rewrite | rephrase | WikiText-2 |
|---|---|---|---|---|
| Qwen2.5-3B | 0.0082 | **1.000** | **0.857**（36/42） | 1.0068 |
| Qwen2.5-7B | 0.0043 | **1.000** | **0.929**（39/42） | 1.0070 |

0.625 属于八目标样本。同样本量下，3B 在 rephrase 上落后 7 个点，每次编辑约花 2 倍相对 delta。

## R8 · Llama-3.1-8B

| 配置 | rel.Δ | rewrite | rephrase | WikiText-2 | 峰值 |
|---|---|---|---|---|---|
| int8，第 31/32 层 | 0.0072 | 1.000 | 0.500 | 1.284 | 10.2 GB |
| int4，第 30/32 层 | 0.0138 | 0.500 | 0.500 | 1.031 | 8.6 GB |

各两次编辑。**int8** 用一个稠密 block 在 10.2 GB 达到 rewrite 1.000；**int4** 用两个稠密 block 在 8.6 GB 把困惑度保持在 1.031。

## R9 · 等预算 LoRA baseline {#r9-equal-budget-lora-baseline}

同一层（20/24）、同三个 prompt、同目标、同 30 步。Qwen2.5-0.5B，float32。

| 方法 | rewrite | rephrase | WikiText-2 | MMLU（n=8） |
|---|---|---|---|---|
| rank-1，30 步 | 1.000 | 0.976 | **1.044** | 0.500 → 0.625 |
| LoRA rank 8，30 步 | 1.000 | **1.000** | **1.441** | 0.500 → **0.250** |

LoRA 表达力更强：它把留出改写推到 1.000。它也花了 46,080 个无约束参数，并让模型付出 44% 困惑度和 25 个 MMLU 点的代价。rank-one 编辑对受保护 key 投影并受预算限制，因此用十分之一的损伤达到几乎相同的有效性。

## 工程记录

| 问题 | 修复 |
|---|---|
| `transformers.Conv1D` 以 `(in, out)` 存储，数学层假设 `(out, in)` | `HFModel.read/write` 归一化方向 |
| GPT-2 tokenizer 没有 pad token | `encode` 手动 padding 并返回 mask |
| padded batch 在 pad 位置评分 | logits 和 NLL 选择最后一个**真实** token |
| 改写从未预测源事实 | 改写集先在 base model 上过滤 |
