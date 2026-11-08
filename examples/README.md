# Examples

Each example is a runnable entry point rather than a notebook fragment. Start
with the zero-download path, then move to a real checkpoint when you need
model-level evidence.

## Zero download

| example | what it demonstrates | command |
|---|---|---|
| `tiny_edit.py` | one edit, two budgets, and a fail-closed verdict | `python examples/tiny_edit.py` |
| `budget_sweep.py` | cumulative-edit budget and measured damage curve | `python examples/budget_sweep.py` |
| `contract_synthesis.py` | admit the protected subset that fits a budget | `python examples/contract_synthesis.py` |
| `longform_check.py` | reference-scored long-form coherence | `python examples/longform_check.py` |

## Real checkpoints

| example | what it demonstrates | command |
|---|---|---|
| `gpt2_rome.py` | GPT-2 factual edit with rewrite, paraphrase, KL, and perplexity gates | `python examples/gpt2_rome.py` |
| `file_edit.py` | streaming safetensors write without loading the whole model twice | `python examples/file_edit.py --help` |
| `bench_edits.py` | zsRE target injection, WikiText-2, and MMLU on open datasets | see below |
| `lora_baseline.py` | equal-step LoRA comparison on the same layer and prompts | `python examples/lora_baseline.py` |

## Reproduce a benchmark panel

```bash
export HF_ENDPOINT=https://hf-mirror.com
python scripts/fetch_benchmarks.py

HF_HUB_OFFLINE=1 python examples/bench_edits.py \
  --model gpt2 --layer 8 --edits 8 --dtype float32 --solver gradient
```

Every loader writes a `.provenance.json` file next to its dataset. The record
contains source, revision, licence, item count, and SHA-256, so a measured panel
can be checked later without trusting a moving URL.

## Small, inspectable outputs

The examples print tables rather than hiding decisions inside a framework:

```text
budget       rel.delta   efficacy     kl_p95   ppl_ratio   verdict
------------------------------------------------------------------
beta=1           0.107       1.00     0.0002       1.001   PASS
beta=12          1.279       1.00     0.0146       2.024   FAIL: ppl_ratio
```

If an example changes a user-visible number, the corresponding documentation
page and reproduction command are updated in the same change.
