# 基准测试

这里的每个数字都来自带许可证和 SHA-256 记录的公开数据集。文件由 `scripts/fetch_benchmarks.py` 下载，并在旁边写入 `.provenance.json`。

## 数据集

| 数据集 | 来源 | 许可证 | 条目 | SHA-256 |
|---|---|---|---|---|
| zsRE（development） | `hf://datasets/wangzn2001/zsre/structured_zeroshot-dev-new_annotated_final.jsonl` | 镜像未声明；上游 zsRE，Levy et al. 2017 | 10,813 | `05bbeb0ed8c4…` |
| WikiText-2 raw（test） | `hf://datasets/Salesforce/wikitext/wikitext-2-raw-v1/test` | CC-BY-SA-3.0 AND GFDL | 4,358 | `5f1bea067869…` |
| MMLU（test） | `hf://datasets/cais/mmlu/all/test` | MIT | 14,042 | `74a41822ce7d…` |

!!! warning "zsRE 许可证"
    这里使用的镜像未声明许可证。zsRE 广泛用于知识编辑文献（MEND、SERAC、MEMIT）；如果你要再分发，请先在原始来源核实条款。

随时校验：

```python
from machaon.bench import read_provenance, verify
verify("~/.cache/machaon/bench/zsre-dev.jsonl")
```

## 协议

**Target injection。** 对每个 zsRE 问题，从数据集自带的实体词表里选一个干扰 token，把模型编辑成预测它，然后测量原问题和人工改写。

为什么不“修复模型答错的条目”？因为 Qwen2.5-0.5B 在 300 个 zsRE 首 token 里答对 299 个，那个协议几乎没有提升空间。Target injection 在弱模型和强模型上都有效。

**WikiText-2 — 困惑度。** 所有编辑前后，在留出文档上计算平均 NLL；报告 `exp(Δ mean NLL)`。

**MMLU — 能力。** 四选一问题，用字母补全的平均 log-probability 评分。

**求解器。** `gradient` 200 步，模型参数冻结，backward 只为 value 向量累积梯度。在 CPU 上运行得到两个发现：

| 观察 | 测量 |
|---|---|
| macOS CPU 上的 bf16 backward | 4.66 秒/步 |
| macOS CPU 上的 float32 backward | **0.18 秒/步**（快 26 倍） |
| Qwen2.5-0.5B 第 12 层，一次编辑 | rel. delta 0.150 |
| 同模型第 20 层 | rel. delta **0.0043**（便宜 35 倍） |

两者都比 solver 超参重要。CPU 上用 float32，并选择最靠近输出、仍能改变行为的层。

## 结果

同协议、同 8 次编辑、同评估器。

| 模型 | 参数 | 层位 | 平均 rel.Δ | rewrite | rephrase（513） | WikiText-2 | MMLU |
|---|---|---|---|---|---|---|---|
| GPT-2 124M | 124M | 8 / 12 | 0.0135 | 0.500 | 0.259 | 1.066 | 0.220 → 0.220（n=100） |
| Qwen2.5-0.5B | 494M | 20 / 24 | 0.0133 | 0.375 | 0.275 | 1.030 | 0.440 → 0.440（n=50） |
| Qwen2.5-1.5B | 1544M | 20 / 28 | **0.0065** | 0.500 | 0.273 | **1.005** | 0.400 → 0.400（n=50） |
| Qwen2.5-3B | 3086M | 34 / 36 | 0.0072 | 0.625 | **0.327** | 1.009 | 0.500 → 0.500（n=30） |

1.5B 用 100 步，3B 用第 34 层、30 步。趋势是重点：从 0.5B 到 3B，同一次 injection 的相对 delta 约减半，困惑度代价从 3.0% 降到 0.9%。3B 检查点分片（2 个文件）；运行通过 `AutoModelForCausalLM` 加载，每层编辑一个 `down_proj`。

### 1.5B 文件级写入

```
source   : 338 tensors, 1543.7M params, 1 shard(s)
target   : model.layers.20.mlp.down_proj.weight [1536, 8960]
delta    : rel 0.0050
wrote    : /tmp/qwen15b_edited3 (2.3s)
untouched: model.embed_tokens.weight identical=True
```

2.9 GB 检查点 2.3 秒重写，一个 tensor 改变，其余逐字节相同。这是为 7–8B 设计的路径：模型无法舒适地和第二份副本一起放进内存。

## 7B 在 32 GB 笔记本上

Qwen2.5-7B 是 7.6B 参数、四个分片的检查点。用 **int8 权重**在 32 GB 机器上运行：常驻约 9.4 GB，每 forward 2.5 秒。

**文件级写入**重写一个分片，其余硬链接：

```
source   : 339 tensors, 7615.6M params, 4 shard(s)
target   : model.layers.24.mlp.down_proj.weight [3584, 18944]
delta    : rel 0.0030
shard    : model-00004-of-00004.safetensors (linked 3, copied 0)
wrote    : /tmp/qwen7b_edit (2.3s)
untouched: lm_head.weight identical=True
```

峰值内存是一个分片（约 4 GB），而不是整个检查点。

**完整评估**在同一 int8 模型上运行：`optimum-quanto` 把它保持在约 9.4 GB，`HFModel.densify_from(layer)` 把编辑层及其后续层转成 float32，因此 backward 只穿过 28 个 block 中的 2 个。

| solver | 平均 rel.Δ | rewrite | rephrase | WikiText-2 | 峰值 RSS |
|---|---|---|---|---|---|
| `direction`（无 backward） | 0.0503 | 0.750 | 0.667 | 1.260 | 10.6 GB |
| **`gradient`，局部** | **0.0043** | **1.000** | **0.929** | **1.007** | 10.2 GB |

配置：28 层中的第 26 层，30 步，lr 0.05，int8 权重且 `lm_head` 不量化，三次编辑，每次 15 个改写。

### 深度到输出

用 7B 设置重跑 3B（36 层中的第 26 层，剩 10 个 block）：

| 3B 设置 | 后续 block | 平均 rel.Δ | rewrite | rephrase | WikiText-2 |
|---|---|---|---|---|---|
| 第 26 层，lr 0.02，80 步 | 10 | 0.0052 | 0.625 | 0.230 | 1.002 |
| 第 26 层，lr 0.05，30 步 | 10 | 0.0079 | 0.500 | 0.236 | 1.002 |
| 第 32 层，lr 0.05，30 步 | 4 | 0.0068 | 0.625 | 0.312 | 1.009 |
| **第 34 层，lr 0.05，30 步** | **2** | 0.0072 | 0.625 | **0.327** | 1.009 |

深度帮助泛化（0.230 → 0.327），但在八个目标上进入平台。

**同目标数的运行修正了读法。** 用与 7B 实验相同的三个目标、42 个改写，3B 达到几乎相同的数字：

| 模型 | 编辑数 | 平均 rel.Δ | rewrite | rephrase | WikiText-2 |
|---|---|---|---|---|---|
| Qwen2.5-3B，第 34 层 | 3 | 0.0082 | **1.000** | **0.857**（36/42） | 1.0068 |
| Qwen2.5-7B，第 26 层（int8） | 3 | 0.0043 | **1.000** | **0.929**（39/42） | 1.0070 |

所以早先的 0.625 是八目标样本的属性——最后五项更难——不是硬容量上限。同样本量下，两个模型主要差在成本：3B 需要约两倍相对 delta 得到同样结果。

## 8B：当前最大实测模型

Llama-3.1-8B 是 8,030.3M 参数、四个分片，在声明的上限内。32 GB 机器上跑两种配置：

| 配置 | 平均 rel.Δ | rewrite | rephrase | WikiText-2 | 峰值 RSS |
|---|---|---|---|---|---|
| int8，第 31/32 层（一个 block 稠密） | 0.0072 | **1.000** | 0.500 | 1.284 | 10.2 GB |
| int4，第 30/32 层（两个 block 稠密） | 0.0138 | 0.500 | 0.500 | **1.031** | 8.6 GB |

各两次编辑，20 个改写，两篇 WikiText 文档，MMLU n=8。

**int8** 用一个稠密 block 在 10.2 GB 达到 rewrite 1.000。**int4** 用两个稠密 block 在 8.6 GB 把困惑度保持在 1.031。

**7.6B int8 加两 block 窗口**达到 rewrite 1.000 / rephrase 0.929 / 困惑度 1.007——目前实测的最强组合。

`--quantize int4` 需要 `ninja` 在 PATH 中，供 quanto 的 unpack kernel 使用。

## 为什么不直接微调？

`examples/lora_baseline.py` 在同一层、同三个 prompt、同目标 token、同 30 步梯度上训练 rank-8 adapter。Qwen2.5-0.5B，float32：

| 方法 | rewrite | rephrase | WikiText-2 | MMLU（n=8） |
|---|---|---|---|---|
| rank-1 闭式，30 步 | 1.000 | 0.976（41/42） | **1.044** | 0.500 → 0.625 |
| LoRA rank 8，30 步 | 1.000 | **1.000** | **1.441** | 0.500 → **0.250** |

LoRA 表达力更强：它把留出改写推到 1.000。它也花掉 46,080 个无约束参数，并把模型一起带走——44% 困惑度和 25 点 MMLU 下降。rank-one 编辑对受保护 key 投影并受预算限制，因此用十分之一的损伤达到几乎相同的有效性。

## 累计损伤

`examples/budget_sweep.py` 一次应用一个编辑，并在检查点测留出困惑度。GPT-2 124M，第 8 层，100 步：

| 编辑数 | 累计 Δ | WikiText-2 困惑度 |
|---|---|---|
| 0 | 0.000 | 1.000 |
| 8 | 0.089 | 1.049 |
| 16 | 0.151 | 1.098 |
| 32 | 0.255 | 1.191 |

`EditBudget(max_total=0.10)` 在第 10 次编辑时拒绝。这是默认预算的经验基础：累计相对变化超过约 0.1 后，124M–0.5B 模型已经可见退化。

## 长文本连贯性

`examples/longform_check.py` 应用 Eiffel Tower 编辑，并比较五个 prompt 上 120 token 的 greedy 生成。连贯性由**未编辑**模型评分。

| 模型 | baseline 重复率 | 编辑后重复率 | Δ | NLL ratio | 空生成 |
|---|---|---|---|---|---|
| GPT-2 124M（120 tok） | 0.621 | 0.578 | **−0.043** | 1.089 | 0.0 |
| Qwen2.5-1.5B（80 tok） | 0.097 | 0.196 | **+0.099** | 0.938 | 0.0 |

GPT-2 上编辑没有改变重复率；1.5B 上它大约翻倍，正好落在 0.10 gate 上。这是警告，不是值得庆祝的通过：阈值可能偏松，80 token 也偏短。

## 复现

```bash
export HF_ENDPOINT=https://hf-mirror.com
python scripts/fetch_benchmarks.py

HF_HUB_OFFLINE=1 python examples/bench_edits.py \
  --model gpt2 --layer 8 --edits 8 --dtype float32 --solver gradient

HF_HUB_OFFLINE=1 python examples/bench_edits.py \
  --model /path/to/Qwen2.5-0.5B --layer 20 --edits 8 \
  --dtype float32 --solver gradient --mmlu 50 --wikitext 20
```

## 下一批测量

1. 每个配置 100+ 次编辑，并给出方差估计。
2. 跨模型的累计预算 sweep。
3. 每个规模上的长文本连贯性。
4. 3B 与 7B 的 LoRA baseline。
