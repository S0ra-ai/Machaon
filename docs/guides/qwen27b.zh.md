# Qwen3.6-27B 权重空间编程

Qwen3.6-27B 是带视觉编码器的多模态检查点。文本解码器有 64 层，LLM 的 MLP
输出投影位于：

```text
model.language_model.layers.<layer>.mlp.down_proj.weight
```

Machaon 会优先识别这个多模态路径，再回退到普通的 `model.layers`。

## 1 · 安装匹配的 runtime

Qwen3.6 需要较新的 Transformers 和 `qwen3_5` 支持。建议使用独立环境：

```bash
uv venv .venv-qwen36
source .venv-qwen36/bin/activate
uv pip install -e ".[hf]"
uv pip install -U "transformers[serving]" torchvision pillow optimum-quanto
```

`machaon.runtime.load_pretrained` 会根据 config 自动选择
`AutoModelForImageTextToText` 或 `AutoModelForCausalLM`。

## 2 · 下载与检查

```bash
python scripts/fetch_model.py \
  --repo Qwen/Qwen3.6-27B \
  --dest ./qwen36-27b \
  --endpoint modelscope
```

先只读 header，不要加载全部权重：

```python
from machaon.adapters import Checkpoint

ckpt = Checkpoint("./qwen36-27b")
print(ckpt.summary())
print(ckpt.find(layer=60, suffix="down_proj.weight"))
```

## 3 · 选择内存工作点

| 模式 | 权重约占用 | 适用 |
|---|---:|---|
| bf16 | ~54 GB | 大内存/GPU |
| int4 | ~14 GB | 34 GB 本机评估 |
| checkpoint-only | 一个 shard（约 4 GB） | 已知 delta 的单张量写入 |

在 34 GB 机器上使用 int4，并把编辑位置放在靠近输出的第 56–60 层：

```python
import torch
from transformers import AutoTokenizer, QuantoConfig
from machaon.runtime import HFModel, load_pretrained

tokenizer = AutoTokenizer.from_pretrained("./qwen36-27b")
model = load_pretrained(
    "./qwen36-27b",
    quantization_config=QuantoConfig(weights="int4"),
    device_map="cpu",
)
runtime = HFModel(model, tokenizer)
assert runtime.layer_path == "model.language_model.layers"
runtime.densify_from(60)
```

## 4 · 求解一次受约束编辑

```python
from machaon.procedures import solve_edit

neutral = [
    "The capital of Germany is",
    "Water freezes at a temperature of",
    "Two plus two equals",
    "The Moon orbits the",
]

proposal = solve_edit(
    runtime,
    "The Eiffel Tower is located in the city of",
    " Rome",
    layer=60,
    preserve_prompts=neutral,
    solver="gradient",
    steps=40,
    lr=0.02,
    max_rel_norm=0.005,
)
print(proposal.module, proposal.relative_delta)
```

第一次也可以使用 `solver="direction"`，它不建反向图，速度更快。

## 5 · 校验与写入

```python
from machaon.adapters import Checkpoint
from machaon.verify import check_protected

W = runtime.read(proposal.module)
W_new = W + proposal.delta
report = check_protected(W, proposal.delta, protected_keys)
assert report["passed"]

ckpt = Checkpoint("./qwen36-27b")
ckpt.write_tensor(proposal.module, W_new, "./qwen36-edited")
```

`write_tensor` 只重写包含目标 tensor 的那个 shard，其余 shard 硬链接，源检查点不变。
这是 27B 模型推荐的保存方式。

## 6 · 用 Session 管理 apply / evaluate / rollback

```python
from machaon.session import Session

session = Session(runtime, "./qwen36-27b")
session.baseline(neutral, held_out_documents)
session.propose(
    "The Eiffel Tower is located in the city of",
    " Rome",
    layer=60,
    preserve_prompts=neutral,
    solver="gradient",
    steps=40,
    lr=0.02,
    max_rel_norm=0.005,
)
session.apply()
metrics = session.evaluate(
    ["The Eiffel Tower is located in the city of"],
    [tokenizer(" Rome", add_special_tokens=False)["input_ids"][0]],
)
session.rollback()
```

完整的 `EditSuite`/`judge` 使用同一模型的 before/after 状态或 checkpoint 快照，
不要同时加载两份 27B 权重。预算从小开始，例如 `0.003–0.01`。

## 7 · 在浏览器里走同一套流程

本地控制台复用同一个 session 和 checkpoint 白名单：

```bash
pip install -e ".[web]"
machaon-web --allowed-root ./qwen36-27b --port 8765
```

打开启动时打印的 URL，先检查 checkpoint header，再用 `dtype=auto`、
`quantize=int4`、`device=cpu`、第 60 层和预算 `0.01` 打开会话。把中性 prompt
作为 baseline，求解 proposal，先读相对变化，再按 **Apply**。证据栏会流式显示
任务状态，并在评估后执行 fail-closed 验收门；rollback 直接恢复原始 tensor 字节。
