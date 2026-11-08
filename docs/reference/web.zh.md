# Web 控制台

本地 Web 控制台是 Python API、CLI 和 MCP 服务之外的第三个薄入口。它把同一个
`machaon.tools` / `machaon.session` 操作呈现给浏览器，而不是复制一套模型逻辑：

```text
浏览器 ──> FastAPI / SSE ──> machaon.tools / machaon.session
                                  │
                    CLI ──────────┤
                    MCP server ───┘
```

## 安装与启动

核心包只依赖 NumPy，需要浏览器控制台时安装可选依赖：

```bash
pip install "machaon[web]"
machaon-web --allowed-root /models --port 8765
```

启动后会打印带一次性 token 的本地 URL。服务默认只绑定 `127.0.0.1`；
可以用 `--allowed-root` 或 `MACHAON_WEB_ALLOWED_ROOTS` 指定 checkpoint 根目录。

## 界面语言

控制台提供 English、简体中文、日本語和한국어。顶栏语言选择器会同步切换静态文案、
任务状态、指标和错误消息，并把选择保存到 `localStorage`。也可以直接用
`?lang=zh`、`?lang=ja`、`?lang=ko` 打开对应语言；没有显式选择时跟随浏览器，
无法识别时回退到 English。

## 三步工作流

1. **Checkpoint**：读取 safetensors 文件或分片目录的 header，显示 tensor 数、
   参数数、dtype 和分片数。浏览器只看到名称、形状和类型，不接收权重值。
2. **编辑合同**：打开模型会话、采集 baseline、选择层数并求解 proposal。
   preserve prompts 会进入零空间约束；量化模型做 gradient solve 前勾选
   **Densify from edit layer**。proposal 在写入前报告相对变化和目标概率。
3. **证据**：应用待处理 proposal，使用 baseline 评估，并查看 fail-closed
   验收门。时间线记录状态、结果和错误；rollback 恢复原始 tensor 字节。
   需要持久化时，把 Applied session 的 delta 写到新 checkpoint 目录，源目录不变。

模型加载、求解和评估在单 worker 中运行，并通过 SSE 推送进度。一个进程同一时间
持有一个模型会话。所有模型级 POST 返回 job id，事件流包含 `status`、`result`
和 `error`。

应用并评估后，用 **Persist** 把当前 delta 写成新 checkpoint：单个 tensor 走
`Checkpoint.write_tensor`，多个 tensor 走 `Checkpoint.write_with`。源目录保持
不变，输出目录必须位于允许根目录内。

## 大模型示例：Qwen3.6-27B

控制台是模型无关的；同一套控件适用于所有受支持架构。27B 只是一个说明大模型
工作点的示例，建议在 34 GB 统一内存机器上使用 `int4` 或 `int8`：

```text
path:      ./qwen36-27b
dtype:     auto
quantize:  int4
device:    cpu
layer:     60
budget β:  0.01
module:    mlp.down_proj
solver:    gradient
densify:   勾选
```

先从 header 检查 checkpoint，避免无意加载全部权重。完整的 Transformers runtime、
分片写入和 `Checkpoint.write_tensor` 见 [Qwen3.6-27B 指南](../guides/qwen27b.md)。

## 安全边界

控制台留在 loopback；把允许根目录收窄到实际使用的 checkpoint 目录；把打印的
token 当作临时能力凭证。Web 层不接受任意 Python 代码，所有写操作都经过路径
白名单、预算检查和事务回滚。需要长期保存时使用 checkpoint 文件级写入操作。
