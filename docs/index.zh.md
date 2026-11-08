# Machaon

**直接对 LLM 做权重手术 —— 改权重，不训梯度。**

*只取箭头，不动身体。*

Machaon 直接修改模型参数，并证明修改没有破坏其他能力：约束是什么、实际改了什么、出错时如何逐字节还原。一次编辑是一份可 diff、可重放、可回滚的内容寻址产物。

## 三条保证

| 保证 | 实现 | 检查 |
|---|---|---|
| 受保护的 key 不动 | 零空间投影，`ΔW k⊥ = 0` | `verify.check_protected` |
| 被拒的编辑完全回退 | 先复制原始字节 | `transaction.rollback` |
| 未测量的编辑不通过 | 缺失指标视为失败 | `hygieia.evaluate` |
| 编辑可复现 | delta 内容寻址 | `store.DeltaStore` |

## 安装

```bash
git clone https://github.com/S0ra-ai/Machaon.git
cd Machaon

uv pip install -e .              # 仅 NumPy：数学、验收门、互操作
uv pip install -e ".[hf]"        # + torch + transformers，跑真实模型
uv pip install -e ".[bench]"     # + pyarrow，跑基准
uv pip install -e ".[mcp]"       # + MCP SDK，Python 3.10+
uv pip install -e ".[web]"       # + FastAPI + uvicorn，本地浏览器控制台
uv pip install -e ".[dev,docs]"  # 测试、lint、覆盖率、文档
```

Python 3.9–3.13 全矩阵测试；`make ci` 一条命令跑完离线检查。

## 30 秒上手

```python
import numpy as np
from machaon import rank1_constrained

W_new, delta = rank1_constrained(W, key, value, preserve=protected, max_rel_norm=0.02)
```

```python
from machaon.verify import check_protected

report = check_protected(W, delta, protected)
assert report["passed"] and report["max_drift"] < 1e-12
```

第二段才是重点：第一段是公式，第二段是判决。

## 实测规模

| 模型 | 层位 | 平均 rel.Δ | rewrite | rephrase | WikiText-2 |
|---|---|---|---|---|---|
| GPT-2 124M | 8 / 12 | 0.0135 | 0.500 | 0.259 | 1.066 |
| Qwen2.5-0.5B | 20 / 24 | 0.0133 | 0.375 | 0.275 | 1.030 |
| Qwen2.5-1.5B | 20 / 28 | **0.0065** | 0.500 | 0.273 | **1.005** |
| Qwen2.5-3B | 34 / 36 | 0.0072 | 0.625 | **0.327** | 1.009 |
| Qwen2.5-7B（int8） | 26 / 28 | **0.0043** | **1.000** | **0.929** | **1.007** |
| Llama-3.1-8B（int4） | 30 / 32 | 0.0138 | 0.500 | 0.500 | 1.031 |

## 文档

- [安装](getting-started/installation.md)
- [快速开始](getting-started/quickstart.md)
- [编辑一个事实](guides/edit-a-fact.md)
- [评估一次编辑](guides/evaluate.md)
- [基准与复现](reference/benchmarks.md)
- [Python API](reference/api.md)
- [MCP 服务](reference/mcp.md)
- [Web 控制台](reference/web.md)

## 状态

Alpha，但核心已经实测。数学、事务、存储、验收门、运行时、safetensors 适配器与 OpenWSP v1 均已实现，并在 Python 3.9–3.13 上测试。真实模型覆盖 GPT-2 124M、Qwen2.5 0.5B / 1.5B / 3B / 7B 与 Llama-3.1-8B；7B/8B 使用 int8/int4 权重和局部稠密窗口。七个 CI job 覆盖 lint、测试矩阵、适配器与覆盖率、MCP 传输、Web 控制台、文档构建、wheel/sdist 安装。下一步见[路线图](roadmap.md)。
