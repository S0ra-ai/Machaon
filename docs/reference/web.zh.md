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
如果缺少 `torch`、`transformers` 或 `safetensors`，页面会显示 banner 和准确的
`pip install "machaon[hf]"` 命令，而不是让第一次打开会话静默失败。

## 界面语言

控制台提供 English、简体中文、日本語和한국어。顶栏语言选择器会同步切换静态文案、
任务状态、指标和错误消息，并把选择保存到 `localStorage`。也可以直接用
`?lang=zh`、`?lang=ja`、`?lang=ko` 打开对应语言；没有显式选择时跟随浏览器，
无法识别时回退到 English。

## 首访引导

第一次通过认证进入时，控制台会打开一页短引导，先指向 **FIRST EDIT（首次编辑）**
零到一卡片，再解释一次编辑在做什么、完整的
“检查 → 打开 → baseline → 求解 → 应用 → 评估 → 保存/回滚”闭环，以及每个按钮
的含义。点击 **开始探索** 会记录已读；顶栏的 **引导** 可以随时重新打开。
Hero 下方的四步提示条会一直保留，作为轻量提醒。

## 首次编辑卡片

首次认证进入时还会显示 **FIRST EDIT（首次编辑）** 卡片。它把已有环节串起来：
没有选择检查点时，可以直接在卡片里选择 starter（默认 SmolLM2-135M，多语言可用
Qwen2.5-0.5B，也可选 GPT-2），下载后打开模型、预填
"The Eiffel Tower is in the city of -> Rome"、运行教学配方并显示结果。两个示例按钮
分别加载“会应用编辑”的 `Eiffel → Rome` 和“模型本来就知道”的 `France → Paris`，
新用户无需输入即可看到两种结果。会话打开后卡片自动消失；通过的编辑保留在 session 中，
按 **保存** 写出。教学成功后，卡片会显示选中的层、相对 Δ、面板保留数和 KL p95，
让第一次成功也带着与高级工作区相同的证据词汇；同时保留一个预填
`<model>-machaon` 的 **检查点输出** 字段和 **保存检查点** 按钮，走与右栏相同的
`/api/session/save`。保存后，**打开已保存检查点** 会关闭当前编辑会话（回滚其内存
字节）、打开保存的检查点，留下一个干净的 session；**完成** 则保留当前会话。打开后，卡片会先用刚才的
prompt/answer 对新检查点自动复验一次，并报告保存后的回答是否仍然复现；随后清空句子、把输出更新为新检查点的
下一个编号路径 `<name>-machaon-N`（例如 `<name>-machaon-2`），并提供 **再教一条事实**，可以继续同一闭环而不必回到主表单。每次保存都会开启新的 session 和基线；
需要跨事实的累计预算与原始面板漂移检查时，请使用 **教一组事实**。
**隐藏** 会把选择保存到 `localStorage`。

最后一次成功打开或保存的检查点路径也会记录在 `localStorage`；刷新页面后会恢复到
Checkpoint 输入框，并让 FIRST EDIT 状态明确显示将使用哪个检查点，不会在用户
不知情时重新下载 starter。

保存后，卡片中的 **教一组事实** 会打开多事实面板，预填当前事实并建议
`<model>-machaon-many` 输出容器。列表结果会展开每条事实的 prompt、状态、层、
Δ 和全局漂移，而不只是状态标签。

## 三步工作流

1. **Checkpoint**：读取 safetensors 文件或分片目录的 header，显示 tensor 数、
   参数数、dtype 和分片数。浏览器只看到名称、形状和类型，不接收权重值。
   没有检查点时，选择一个 starter 并点 **Download starter**：它会从允许根目录
   下载并尝试镜像，填好路径，不会上传任何内容。任务横幅会通过 SSE 显示当前
   端点、文件和字节/百分比进度。勾选 **下载后自动打开** 会直接开启会话，下一步
   就是 **ONE SENTENCE** 输入框。
2. **编辑合同**：打开模型会话、采集 baseline、选择层数并求解 proposal。
   preserve prompts 会进入零空间约束；量化模型做 gradient solve 前勾选
   **Densify from edit layer**。proposal 在写入前报告相对变化和目标概率。
3. **证据**：应用待处理 proposal，使用 baseline 评估，并查看 fail-closed
   验收门。时间线记录状态、结果和错误；rollback 恢复原始 tensor 字节。
   需要持久化时，把 Applied session 的 delta 写到新 checkpoint 目录，源目录不变。

### 一句话教学

编辑栏顶部的 **ONE SENTENCE**（一句话）区域是最短路径。它接受与
`machaon intent` 相同的句式，并运行与 `machaon teach` 相同的教学配方：

1. **预览解析** 会编译句子，并填入 prompt/answer 字段。
2. **自动教学** 会搜索一段深度窗口，把编辑投影到 preserve prompts 的零空间，
   应用最佳候选；只有目标、预算、面板和 KL 检查都通过才保留。被拒绝的候选会
   自动回滚。

可选的 **同时写出编辑后的检查点** 字段会把通过的编辑保存到允许根目录下的新
目录。留空则编辑只保留在 session 中，之后仍可用 **Save** 保存。

**教一组事实** 也在同一个区域：每行一句，系统逐条运行完整搜索，并把
`<out>/final` 与 `<out>/report.json` 写入输出容器。当前 session 仍停留在基础
检查点；**使用最终检查点** 会打开最后一条全局安全结果。

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
