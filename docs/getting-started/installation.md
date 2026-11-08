# Installation

## Requirements

| | |
|---|---|
| Python | 3.9, 3.10, 3.11, 3.12 or 3.13 |
| Core | NumPy only |
| Real models | PyTorch + transformers + safetensors |
| Benchmarks | pyarrow |
| MCP server | `mcp` SDK, Python 3.10+ |

## Install

```bash
git clone https://github.com/S0ra-ai/Machaon.git
cd Machaon
uv venv && uv pip install -e ".[dev]"
```

Extras:

```bash
uv pip install -e ".[hf]"      # torch + transformers + safetensors
uv pip install -e ".[bench]"   # pyarrow, for MMLU / WikiText loaders
uv pip install -e ".[mcp]"     # MCP SDK, Python 3.10+
uv pip install -e ".[dev,docs]"  # tests, lint, coverage, docs site
```

## Verify

```bash
./scripts/ci.sh
```

That runs the tests, the synthetic demo, spec validation, both gate cases, and
lint. It needs no network and no GPU.

Expected tail:

```
Ran 214 tests ...
ALL CHECKS PASSED
```

## Model downloads

If `huggingface.co` is unreachable, point the client at a mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Download a model through whichever endpoint reaches you:

```bash
python scripts/fetch_model.py --repo Qwen/Qwen2.5-1.5B --dest ./qwen15b
```

It tries Hugging Face, the mirror, then ModelScope, and checks the safetensors
headers afterwards.

Benchmark files are fetched separately:

```bash
python scripts/fetch_benchmarks.py
```

They land in `$MACHAON_BENCH` (default `~/.cache/machaon/bench`), each with a
`.provenance.json` recording source, revision, licence, item count and SHA-256.

## Troubleshooting

**`RuntimeWarning: divide by zero encountered in matmul` on macOS** — an
Accelerate FP-flag quirk. Machaon suppresses it inside its own kernels; if you
see it in your code, it is the platform, not the maths.

**`this runtime has no tokenizer`** — you called a text method on an ID-based
runtime. Pass token ids, or construct `HFModel(model, tokenizer)`.
