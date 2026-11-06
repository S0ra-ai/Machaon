# 常见问题

**这是微调吗？**
不是。模型权重上没有 optimizer，也没有训练数据。一次编辑是应用一次的闭式或低秩 delta。唯一的梯度出现在 ROME 求解时，针对单个 value 向量。

**除 GPT-2 外还支持哪些模型？**
运行时能发现 Llama/Qwen/Mistral/Gemma（`model.layers`）、GPT-2/GPT-J（`transformer.h`）、GPT-NeoX（`gpt_neox.layers`）、OPT 和 MPT 的层。两种权重方向都处理。真实检查点覆盖 GPT-2 124M、Qwen2.5 0.5B / 1.5B / 3B / 7B 和 Llama-3.1-8B，每一级使用同一套基准面板。

**为什么不直接用 LoRA？**
同层、同步数实测：rank-8 adapter 达到 rewrite 1.000 / rephrase 1.000，但代价是 44% 困惑度和 25 个 MMLU 点；rank-one 编辑达到 1.000 / 0.976，代价 4% 困惑度且无可测损失。LoRA 表达力更强，也更不受约束。见 [R9](RESULTS.md#r9-equal-budget-lora-baseline)。

**能编辑 70B 模型吗？**
Machaon 的实测范围到 8B：int8/int4 权重加局部稠密窗口，求解保持在 10.2 GB 内。更大的模型可以走检查点级路径，一次流式处理一个 tensor；评估路径随你给的内存扩展。求解成本取决于编辑点之后的窗口，而不是整个模型。

**rank-one 编辑够用吗？**
对单个事实通常够。对行为，低秩加方向（`bake_direction` 或 `ablate_output`）通常更好。批量编辑的下一步是 MEMIT 风格的批量求解，见[路线图](roadmap.md)；今天可以在一个 `EditBudget` 下组合逐事实编辑，或使用低秩方向。

**为什么编辑成功但改写失败？**
key 绑定在单一 token 序列上。从多个改写捕获 key 并取平均；改写集也必须是 base model 本来就能正确回答的，否则你测的是噪声。

**为什么 `rank1_constrained` 报 "preserved subspace"？**
target key 落在受保护 key 的张成空间里，两个约束不兼容。缩小受保护集合，或换一个 key。

**为什么报告 `kl_p95` 而不是均值？**
一个受损 prompt 比一百个未受损 prompt 更重要。均值会把它藏起来。

**为什么用 target injection 而不是修错答案？**
因为 Qwen2.5-0.5B 在 300 个 zsRE 首 token 里答对 299 个。只编辑模型答错的条目，在更强模型上几乎没有提升空间。injection 在强弱两端都有效。

**为什么用 float32 而不是 bf16？**
在 CPU 上，PyTorch 的 bf16 backward 慢得病态：同一个 0.5B 模型上 4.66 秒/步 vs float32 的 0.18 秒/步。delta 两种情况下都用 float64 计算；模型 dtype 只影响求解 value 时的 forward/backward。

**为什么层位影响这么大？**
第 12 层的编辑要穿过后面十二个 block。Qwen2.5-0.5B 上同一个编辑在第 12 层 rel. Δ 是 0.150，在第 20 层是 0.0043——35 倍差异。

**我所在的地方访问不了 Hugging Face。**
用 `python scripts/fetch_model.py --repo ... --dest ...`，它会回退到 ModelScope；或设置 `HF_ENDPOINT=https://hf-mirror.com`。脚本还会在传输后校验 safetensors header。

**为什么重复率相对 baseline 报告？**
因为 GPT-2 在你动手之前就会在 greedy decoding 下循环。绝对阈值会把未编辑模型也判失败，那说明不了编辑的问题。

**能叠加多少次编辑？**
实测。GPT-2 上 32 次 rank-one 编辑（累计 Δ 0.255）带来 19% 困惑度代价；8 次是 5%。`EditBudget(max_total=0.10)` 大约在第 10 次拒绝，是推荐默认值。

**为什么缺指标时 gate 失败？**
否则通过验收最简单的方法就是停止测量。失败关闭是设计选择，不是疏忽。

**能不用 OpenWSP 而只用它的格式吗？**
可以。`machaon.interop.openwsp` 是独立实现，并对照参考实现生成的 fixture 做过一致性测试。

**这会移除安全对齐吗？**
方向消融可以一行做到。因此项目自带拒绝规则、追加式编辑记录和验收门。见 [SECURITY.md](https://github.com/S0ra-ai/Machaon/blob/main/SECURITY.md)。
