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
machaon web --allowed-root /models --port 8765
# machaon-web 是同一个启动器
```

启动后会打印带一次性 token 的本地 URL。服务默认只绑定 `127.0.0.1`；
可以用 `--allowed-root` 或 `MACHAON_WEB_ALLOWED_ROOTS` 指定 checkpoint 根目录。
如果缺少 `torch`、`transformers` 或 `safetensors`，页面会显示 banner 和准确的
`pip install "machaon[hf]"` 命令，而不是让第一次打开会话静默失败。

## 界面

控制台只有一个页面：顶部是检查点栏，下面是 **教一句事实** 输入区，以及
**模型**、**编辑**、**权重程序**、**结果** 四个原生折叠区。没有首访弹窗，
也没有单独的 hero；操作顺序就是页面顺序。

### 检查点栏

粘贴检查点目录或单个 `.safetensors` 文件，按 **检查**。**浏览** 会列出允许根目录
下发现的检查点。成功打开或保存后，路径会记在 `localStorage`，刷新后恢复。

### 教一句事实

这是主路径。输入一句话：

```text
The Eiffel Tower is in the city of -> Rome
```

没有检查点时选择起始模型；也可以选择 CPU、自动、Apple MPS 或 CUDA，然后按
**教学**。卡片使用与 `machaon intent` 相同的语法解析句子，打开所选检查点，
必要时下载起始模型，运行教学配方，并显示与 CLI 相同的目标、预算、面板和 KL
证据。被拒绝的编辑会回滚。

应用成功后出现输出路径和 **保存**。保存会写出新检查点；**打开已保存** 会关闭
当前编辑会话、打开保存后的字节，并先复验答案再清空输入。已经有会话时，同一张
卡片会直接在当前模型上教学；起始模型和设备选择器在不再影响运行时隐藏。

**教一组事实** 启动累计多事实链。**创建程序** 把当前事实变成可编辑的 `.machaon`
程序。两者在成功编辑后仍然可用。

### 模型

**模型** 折叠区包含 header 检查、tensor 名称/形状/类型、可选的起始模型下载、
会话 dtype/设备/预算设置、基线采集，以及不改权重的单事实验证。下载通过 SSE 显示端点、
文件和字节进度；**下载后打开** 会在下载完成后开启会话。

**验证事实** 会重新加载选中的检查点，评估一个 prompt/answer 或每行一条事实，
并显示 PASS/FAIL、答案和准确率。它调用 `/api/check`，不会修改会话。

### 编辑

**编辑** 是精确合同编辑器：prompt、target、保护提示词、提案、应用、评估和回滚。提案是 dry run；**应用** 在事务内写入并计入累计预算。验收失败或回滚都会
恢复原始字节。

**高级编辑** 展开层、solver、步数、学习率、模块后缀、相对预算和 densify 设置。
**教一组事实** 为每条事实运行一次完整搜索，使用累计预算和原始面板漂移检查。
**批量编辑** 接受多条 prompt 和激活行，报告 exact/clamped、误差、秩和逐 key 误差。

### 权重程序

**权重程序** 折叠区接受与 `machaon program` 相同的 `.machaon` 文本或 JSON。
**检查** 调用 `/api/program/plan`，在不加载模型的情况下解析路径、事实、预算、
设备和输出。**运行** 以后台任务执行整条链，写出
`<output>/program-report.json`，重新加载保存的检查点并验证每条事实后才接受。

**更多** 包含程序目录和报告路径、从当前检查点预填、本地 `.machaon` 导入、
SmolLM2 / 两个事实 / 双语 Qwen 模板、最近程序和运行、事实列表生成器以及报告验证。
**验证报告** 检查 SHA-256 和结构；**验证报告 + 模型** 还会重新加载检查点并重跑
事实。可选复选框可以要求当前编辑器程序匹配，或改用当前选中的检查点。相对路径
和所有浏览器文件访问都受同一白名单约束。

### 结果

**结果** 折叠区包含最近的提案、fail-closed 验收结论、检查点保存、WSP 控件和事务
活动日志。WSP 支持报告审计、会话评估、保存证据、导出 OpenWSP 和验证绑定。缺失
验收指标仍然按失败处理。

### 本地状态与语言

控制台提供 English、简体中文、日本語、한국어。顶部选择器会同时更新静态文案和
动态任务/证据信息，选择保存在 `localStorage`，也支持 `?lang=zh`、`?lang=ja`、
`?lang=ko` 直达。最近程序和运行只保存在本地，不会上传。

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
