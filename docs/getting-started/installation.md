# Installation

## Requirements

| | |
|---|---|
| Python | 3.9, 3.10, 3.11, 3.12 or 3.13 |
| Core | NumPy only |
| Real models | PyTorch + transformers + safetensors |
| Benchmarks | pyarrow |
| MCP server | `mcp` SDK, Python 3.10+ |
| Browser console | FastAPI + uvicorn |

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
uv pip install -e ".[web]"     # local browser console
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
Ran 508 tests ...
ALL CHECKS PASSED
```

## Model downloads

`machaon fetch` downloads a safetensors checkpoint without loading it. It tries
Hugging Face, `hf-mirror`, then ModelScope, validates every safetensors header,
and resumes existing files.

```bash
machaon fetch --list
machaon fetch --starter qwen05b --dest ./models/Qwen2.5-0.5B
machaon fetch --repo Qwen/Qwen2.5-1.5B --dest ./qwen15b
```

`HF_ENDPOINT` still applies to the Transformers loader; the built-in fetch has
its own fallback. A repository with no `.safetensors` weights is refused rather
than left as a half-downloaded directory.

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
