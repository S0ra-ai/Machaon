# 安装

## 环境要求

| | |
|---|---|
| Python | 3.9、3.10、3.11、3.12 或 3.13 |
| 核心 | 仅 NumPy |
| 真实模型 | PyTorch + transformers + safetensors |
| 基准测试 | pyarrow |
| MCP 服务 | `mcp` SDK，Python 3.10+ |
| 浏览器控制台 | FastAPI + uvicorn |

## 安装

```bash
git clone https://github.com/S0ra-ai/Machaon.git
cd Machaon
uv venv && uv pip install -e ".[dev]"
```

可选依赖：

```bash
uv pip install -e ".[hf]"       # torch + transformers + safetensors
uv pip install -e ".[bench]"    # pyarrow，MMLU / WikiText 加载器
uv pip install -e ".[mcp]"      # MCP SDK，Python 3.10+
uv pip install -e ".[web]"      # 本地浏览器控制台
uv pip install -e ".[dev,docs]" # 测试、lint、覆盖率、文档站
```

## 验证

```bash
./scripts/ci.sh
```

它会运行测试、合成 demo、spec 校验、两个 gate 用例和 lint。不需要网络，也不需要 GPU。

预期结尾：

```
Ran 225 tests ...
ALL CHECKS PASSED
```

## 下载模型

如果 `huggingface.co` 不可达，把客户端指向镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

通过可用端点下载模型：

```bash
python scripts/fetch_model.py --repo Qwen/Qwen2.5-1.5B --dest ./qwen15b
```

脚本依次尝试 Hugging Face、镜像、ModelScope，并在传输后校验 safetensors header。

基准数据单独下载：

```bash
python scripts/fetch_benchmarks.py
```

文件落在 `$MACHAON_BENCH`（默认 `~/.cache/machaon/bench`），每个文件旁边都有 `.provenance.json`，记录来源、revision、许可证、条目数和 SHA-256。

## 常见问题

**macOS 上出现 `RuntimeWarning: divide by zero encountered in matmul`** —— Accelerate 的浮点标志怪癖。Machaon 在自己的 kernel 内抑制了它；如果你在自己的代码里看到，那是平台行为，不是数学错误。

**`this runtime has no tokenizer`** —— 你在一个只接受 token id 的 runtime 上调用了文本方法。传 token id，或构造 `HFModel(model, tokenizer)`。
