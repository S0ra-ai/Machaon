# MCP 服务

Machaon 通过 Model Context Protocol 暴露文件级操作：agent 可以检查检查点、审计别人的编辑、应用 delta，并对结果执行验收门——不需要加载模型。

```bash
pip install "machaon[mcp]"
python -m machaon.mcp_server
```

同时支持两代 SDK：`mcp` 1.x 暴露 `FastMCP`，2.x 改名为 `MCPServer`；装饰器和 `run(transport="stdio")` 表面一致，`create_server()` 会选择已安装的那一个。

CI 会把服务作为子进程启动，并通过真实客户端调用工具（`scripts/mcp_smoke.py`），因此传输层坏掉会直接让构建失败。

## 客户端配置

Codex：

```bash
codex mcp add machaon -- python -m machaon.mcp_server
```

Claude Desktop（`claude_desktop_config.json`）：

```json
{
  "mcpServers": {
    "machaon": {
      "command": "python",
      "args": ["-m", "machaon.mcp_server"]
    }
  }
}
```

文件级服务走 stdio，不持有模型状态，因此重启成本很低。模型级工具共享一个显式打开的 `Session`。

## 文件级工具

| 工具 | 作用 |
|---|---|
| `describe_checkpoint(path)` | tensor 数、参数量、dtype 混合、分片数——只读 header |
| `list_tensors(path, contains, suffix, layer, limit)` | 名称、形状、dtype；不返回值 |
| `audit_edit(base_path, edit_dir, max_rel_norm)` | 对照 base 校验 OpenWSP v1 编辑并重算漂移 |
| `apply_delta(base_path, tensor, delta_path, out_dir)` | 把一个 `.npy` delta 应用到一个 tensor，写出新检查点 |
| `gate(metrics)` | 应用验收门；缺失指标视为失败 |

每个工具都委托给 `machaon.tools` 中的普通函数，测试直接覆盖这些函数。传输层不增加行为。

## 模型级工具

这些工具作用于一个已加载的 session。`open_model` 是唯一昂贵的调用。

| 工具 | 作用 |
|---|---|
| `open_model(path, dtype)` | 把模型加载进 session |
| `model_status()` | 参数、层数、已编辑 tensor、预算使用 |
| `capture_baseline(neutral_prompts, corpus_texts)` | **编辑前**记录参考分布 |
| `propose_edit(prompt, target, layer, solver, steps, max_rel_norm)` | 求解编辑并报告代价，不写入 |
| `apply_edit()` | 写入待定提案，原子扣预算 |
| `evaluate(prompts, targets)` | 准确率、KL、相对 baseline 的困惑度 |
| `rollback()` | 还原每个已编辑 tensor |

预期循环正是 `machaon.session` 中的那一个：

```
open_model -> capture_baseline -> propose_edit -> apply_edit -> evaluate
                                                        |
                                                    rollback
```

`propose_edit` 在写入前返回相对 delta，这正是重点。agent 可以先判断编辑是否太贵，再决定不应用。

## 错误形状

Session 工具返回 `{"error": "no model is open; call open_model first"}`，而不是抛异常，这样 agent 看到的是原因，而不是笼统的 `Error executing tool`。

## 设计边界

- **文件级工具 header 优先。** 检查、审计、重写检查点都不需要把模型加载进内存。
- **模型级工具显式。** `open_model` 是唯一昂贵调用；`propose_edit` 在 `apply_edit` 写入前报告代价。
- **编辑位置由 agent 决定。** 选层属于 agent 或 `machaon.procedures.rome` 之类的 procedure；服务负责事务和预算。
- **路径使用进程权限。** 在与 shell 相同的信任边界内运行，详见
  [SECURITY.md](https://github.com/S0ra-ai/Machaon/blob/main/SECURITY.md)。
