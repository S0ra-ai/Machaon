# 评测标准 v1（唯一口径）

本文件定义 Machaon 全部跨模型评测的唯一口径。**任何新增数字，只要进论文主表或对外报告，必须按本文件执行；与旧结果的差异在迁就表里显式登记，不静默覆盖。**

版本：v1 · 2026-11-17
单一事实源：`machaon/hygieia/gates.py`（阈值）、`paper/experiments/gpt2_independent_edits.py`（运行参数）

---

## 1. 模型矩阵：四组，职责不同

"什么模型对什么模型"要先区分子集。四组不可混在一张表里比较。

### 组 I · 主对照（等功效，出结论）

| 角色 | 模型 | 参数 | 层数 |
|---|---|---|---|
| A | `gpt2` | 124.4M | 12 |
| B | `Qwen/Qwen2.5-0.5B` | 494M | 24 |

同一协议、同 50 个独立 edit、同层位口径。**论文中"跨架构"的结论只从这一组出。** 另有 `Qwen2.5-0.5B-Instruct` 敏感性面板（20 edit）作为 B 的指令微调对照，不进主表。

### 组 II · 规模阶梯（趋势观察，不等功效）

`Qwen2.5-0.5B / 1.5B / 3B / 7B`。同家族、同协议、同层位口径，每档 50 edit。

**这组只能陈述趋势，不能当等功效估计。** 论文 `journal_selection.md` 已明确要求：3B/7B/8B 行是 scaling observations。

### 组 III · 跨家族抽验

`meta-llama/Llama-3.1-8B`（8.03B，32 层），20 edit。唯一目的是证明结论不是 Qwen 架构特有，**不承担显著性主张**。

### 组 IV · 同模型内部路径对照（真正的 baseline）

不是换模型，是同一份权重换路径：

| 路径 | 设置 | 用途 |
|---|---|---|
| 未编辑 base | 同 checkpoint | 所有指标的比照原点 |
| rank-one（本文方法） | 见 §2 | 主方法 |
| LoRA rank-8 | 同层、同 3 目标、同 30 步 | 等预算表达力上限对照（R9 口径） |
| int8 / int4 | 同 checkpoint 不同量化 | **量化条件必须单独成表，禁止与 fp32 混排** |

---

## 2. 统一运行参数

以下每项对所有模型取同值。"旧值"列是当前仓库里实际用过、需要废弃的值。

| 参数 | 统一取值 | 旧值现状 | 说明 |
|---|---|---|---|
| 编辑目标张量 | `mlp.down_proj.weight` | 一致 | GPT-2 为 `transformer.h.{L}.mlp.c_proj.weight`（Conv1D `(in,out)`，kernel 已归一化） |
| 层位口径 | **相对深度 0.83** | 0.667 / 0.833 / 0.714 / 0.944 / 0.929 / 0.937 | 见 §3 |
| edit 数 / 模型 | 50 | 8 / 8 / 8 / 8 / 3 / 2 | 组 III 可降到 20，须在表注标明 |
| 每 target 改写数 | **15**（`--rephrase-limit 15`） | 64 / 64 / 64 / 64 / 15 / 20 | 分母统一为 `edits × 15`，缺项在脚注报 |
| 求解器 | Adam | 一致 | 只解一个 value 向量，不动模型权重 |
| 步数 / 学习率 | `--steps 100 --lr 0.05` | 200/0.02、100、80、30/0.05、100 | 旧值四套混用 |
| 保护 key 数 | 8 | 一致 | 零空间投影 |
| 预算上限 | `EditBudget(max_total=0.10)` | 一致 | commit 时记账，超支拒绝 |
| WikiText-2 段数 | `--wikitext 20` | 20 / 40 | `bench_edits.py` 默认 40，须显式覆盖 |
| MMLU 题数 | `n=100` | 100 / 50 / 50 / 30 / 8 | 统一到最小可信样本量 |
| 解码 | greedy（temperature 0） | 混用 | 长文本生成长度固定 80 token |
| seed | `sha256(edit_id)[:8] % 2^31` | 隐含 | 每个 edit 独立、可复算，禁止全局随机 |
| dtype | ≤1.5B `float32`；3B `float32`；7B/8B `int8` + 稠密化编辑层及其后 | 见 §4 | 同表内必须一致 |

### 记录要求（缺一项该行作废）

- base checkpoint 的 revision hash
- 数据集 hash（WikiText-2、MMLU、改写集）
- 运行脚本路径 + git commit
- `--edits / --rephrase-limit / --steps / --lr / --dtype` 实际传参
- 输出的 `results/*.json` 原始文件，不手改

---

## 3. 层位：为什么必须按相对深度对齐

现行六行的相对深度完全不同：

| 模型 | 现行层 | 相对深度 | 统一后（0.83） |
|---|---|---|---|
| GPT-2 124M | 8 / 12 | 0.667 | **10** |
| Qwen2.5-0.5B | 20 / 24 | 0.833 | 20（不变） |
| Qwen2.5-1.5B | 20 / 28 | 0.714 | **23** |
| Qwen2.5-3B | 34 / 36 | 0.944 | **30** |
| Qwen2.5-7B | 26 / 28 | 0.929 | **23** |
| Llama-3.1-8B | 30 / 32 | 0.937 | **27** |

层位是 rel.Δ 的一阶敏感量：Qwen 第 12 层 rel.Δ 0.150，第 20 层 0.0043，**差 35 倍**。六行各自取"效果最好的那层"再横向比较，等于给每个模型单独调参后比成绩，属双重标准。

**规则：**

1. 主表一律用相对深度 0.83 取整数层，不逐模型手挑。
2. 每模型的深度扫描（如 R7 对 3B 的 26/30/32/34）保留为附录证据，**不得回流成主表取值**。
3. 表注须同时给出绝对层号与相对深度。

**迁移影响：** 按此口径重跑后，GPT-2 / 1.5B / 3B / 7B / 8B 五行的现有数字全部作废替换。R7 中"深度帮助泛化（0.230→0.327）"的结论改为附录陈述，不再隐含在主表里。

---

## 4. 量化：分表，不合并

| 档位 | 适用 | 依据 |
|---|---|---|
| `float32` | ≤3B | 3B 单副本峰值 11.7 GB；同模型 bf16 backward 比 fp32 慢 26 倍（4.66s vs 0.18s/step），**bf16 禁用** |
| `int8` + 稠密化编辑层及其后 | 7B / 8B | 7B 峰值 10.2 GB；Llama-8B 10.2 GB，rewrite 1.000 |
| `int4` | 仅内存受限时 | Llama-8B 8.6 GB，但 rewrite 从 1.000 掉到 0.500 |

int4 与 int8 的差异足以翻转结论（rewrite 1.000 vs 0.500），因此**任何一行是 int4，该表必须整表标注量化档位，且不得与 fp32/int8 行并列比较**。

---

## 5. 验收门（gate）

单一事实源：`machaon/hygieia/gates.py::GATES`。八项，fail-closed：指标缺失即失败。

| 门 | 判据 | 含义 |
|---|---|---|
| `rewrite_acc` | ≥ 0.95 | 编辑是否落地 |
| `paraphrase_acc` | ≥ 0.60 | 是否泛化到改写问法 |
| `unrelated_acc` | ≥ 1.00 | 无关事实是否纹丝不动 |
| `kl_p95` | ≤ 0.05 | 分布漂移（95 分位） |
| `kl_max` | ≤ 0.50 | 分布漂移（最坏项） |
| `mmlu_delta` | ≥ −0.01 | 能力不退化 |
| `ppl_ratio` | ≤ 1.02 | 语言建模不掉 |
| `longform_contradictions` | == 0 | 长文本不自相矛盾 |

### 与旧文档的冲突（须修正，勿沿用）

| 门 | 代码（准） | `RESULTS.zh.md` R1（误） | R5 长文本（另一套） |
|---|---|---|---|
| `rewrite_acc` | ≥0.95 | ≥0.99 | — |
| `paraphrase_acc` | ≥0.60 | ≥0.50 | — |
| `kl_p95` | ≤0.05 | ≤0.50 | — |
| `ppl_ratio` | ≤1.02 | ≤1.05 | — |
| 长文本 | 矛盾数 ==0 | — | 重复率 ≤0.10、NLL ratio、空生成 0 |

**处置：** 以代码为准；`RESULTS.zh.md` 的 R1 gate 列与 R5 三项回填为代码口径，或显式改名标注为"该项运行的容忍区间，非验收门"。二者不得共用"gate"一词。

### gate 只用于验收判决，不用于 benchmark 表

这是两件事，必须分开：

| 产物 | 判定 | 说明 |
|---|---|---|
| 验收判决 | 过/不过 8 门 | 单次编辑能否 commit。R1 属此类（1.00/1.00/0.00061/1.0008，八门全过） |
| benchmark 表 | 只报指标 | 跨模型对比。R2–R9 属此类，**不做 gate 判决** |

注意 R2–R9 的 paraphrase 值（0.259 / 0.275 / 0.273 / 0.327 / 0.929 / 0.500）本身不满足 gate 的 ≥0.60，因为它们**本来就不是验收判决**，是随模型容量变化的观测值。把两者混谈会得出"多数运行 fail"的错误结论。

---

## 6. 运行模板

```bash
# 组 I / II 单模型（只改 --model / --layers / --quantize，其余锁死）
HF_HUB_OFFLINE=1 python paper/experiments/gpt2_independent_edits.py \
  --model gpt2 \
  --layers 10 \
  --edits 50 \
  --rephrase-limit 15 \
  --steps 100 \
  --lr 0.05 \
  --wikitext 20 \
  --out-dir results/v1/gpt2

# 验收判决（单次编辑，八门）
python examples/bench_edits.py --model gpt2 --layer 10 --edits 1 \
  --steps 100 --lr 0.05 --wikitext 20 --mmlu 100
```

不同模型的 `--layers` / `--model` / `--quantize`：

```bash
--model gpt2                        --layers 10
--model Qwen/Qwen2.5-0.5B           --layers 20
--model Qwen/Qwen2.5-1.5B           --layers 23
--model Qwen/Qwen2.5-3B             --layers 30
--model Qwen/Qwen2.5-7B   --quantize int8 --layers 23
--model meta-llama/Llama-3.1-8B     --quantize int8 --layers 27
```

---

## 7. 待决事项

1. **层位 0.83 的代价**：3B 从 34 层退到 30 层，rephrase 预计从 0.327 降到约 0.312（R7 第 32 层实测），22 个点的深度收益会从主表消失。是否接受，需拍板。
2. **MMLU n=100 的算力**：7B 单次前向 2.5 s，100 题 × 50 edit ≈ 3.5 小时纯评估。若预算不够，降至 n=50 并整表统一，不得逐行不同。
3. **`RESULTS.zh.md` 回填**：R1 gate 列与 R5 三项需改，改完与代码对齐。
4. **组 III 是否要求 50 edit**：Llama-8B 在 int8 下 50 edit 约 5 小时，若只作抽验可保留 20，但表注必须写"20 edit，不等功效"。
