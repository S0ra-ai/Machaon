# 路线图

按“让权重手术更有用”排序，而不是按“最容易发布”排序。每个完成项背后都有测试或一次实测。

## 已交付

- [x] 带可推导不变量的线性代数原语
- [x] 逐字节回滚的事务式应用
- [x] 内容寻址、读取校验的 delta 存储
- [x] 失败关闭的验收门、KL 分位数、bootstrap 区间
- [x] OpenWSP v1 编辑格式，逐字节一致，有一致性测试
- [x] safetensors 适配器：单文件、目录、分片 index
- [x] HF 运行时：架构无关的层发现、方向归一化、padding 正确评分
- [x] 单函数 ROME procedure
- [x] 带 provenance 的基准加载器（zsRE、WikiText-2、MMLU）
- [x] 长文本连贯性套件，由未编辑模型评分
- [x] 带实测成本曲线的累计编辑预算
- [x] Qwen2.5 0.5B、1.5B、3B、7B 各级
- [x] Llama-3.1-8B 的 int8 与 int4 两条路径
- [x] 内存友好的梯度求解（`densify_from`）
- [x] 六个架构族从 config 实例化并测试
- [x] MCP 服务：5 个文件级工具 + 7 个模型级工具，CI 中走真实 stdio
- [x] 基于 FastAPI/SSE 的本地浏览器控制台：token、路径白名单、检查点检查、
      会话编辑、证据与回滚
- [x] 中、英、日、韩四语文档站，自动部署到 GitHub Pages
- [x] 发布流程：sdist、wheel、GitHub release、可选 PyPI 发布

## 现在

- [ ] **MEMIT 风格批量求解。** 多个事实、一个约束系统、一个预算。单编辑路径是积木。
- [ ] **L1 动作空间：沙箱化 Python 编辑函数。** 在 tensor 上声明的函数，用同一套事务和验收门执行。
- [ ] **更多实测层级。** 13B/14B 检查点路径，以及可复现的内存/吞吐表。
- [ ] **更深的长文本套件。** 多轮 prompt、实体一致性、1k+ token 的矛盾检测。

## 下一步

- [ ] 由评估框架驱动的选层搜索
- [ ] OpenWSP writer 的稀疏与结构化编辑形式
- [ ] 可复现编辑运行的公开排行榜
- [ ] Apple Silicon 与 CUDA 原生基准矩阵

## 相邻工具

Machaon 的单位是对现有模型的一次定向编辑。附近的工作有成熟工具，它们的产物可以和本项目的契约组合：

- **合并与 task arithmetic：** mergekit delta 可以通过 `machaon.tools` 和 `machaon.verify` 审计并应用。
- **剪枝与重连：** 结构编辑以 delta 形式到达，通过同一套预算和验收门。
- **微调与 LoRA：** 等预算 LoRA baseline 已在 [`RESULTS.md`](RESULTS.md#r9-equal-budget-lora-baseline) 中实测，adapter delta 可以与 rank-one 编辑对比。
- **推理服务：** 用 `Checkpoint.write_with` 导出编辑后的检查点，交给任何 Hugging Face 兼容栈服务。
