# Qwen3.6-27B weight-space programming

This guide targets the **Qwen3.6-27B** multimodal checkpoint and the same edit
path used by the smaller Qwen models. The text decoder has 64 layers; the edit
target is the MLP output projection at
`model.language_model.layers.<layer>.mlp.down_proj.weight`.

## 1 · Install the matching runtime

Qwen3.6 uses the `qwen3_5` multimodal wrapper. Use the current Transformers
release in an isolated environment, then install the Hugging Face extras:

```bash
uv venv .venv-qwen36
source .venv-qwen36/bin/activate
uv pip install -e ".[hf]"
uv pip install -U "transformers[serving]" torchvision pillow optimum-quanto
```

`machaon.runtime.load_pretrained` selects `AutoModelForImageTextToText` when the
config exposes a vision encoder or a `ConditionalGeneration` architecture. For
text-only causal models it keeps the ordinary `AutoModelForCausalLM` path.

## 2 · Fetch and inspect the checkpoint

ModelScope is usually the fastest endpoint from mainland China:

```bash
python scripts/fetch_model.py \
  --repo Qwen/Qwen3.6-27B \
  --dest ./qwen36-27b \
  --endpoint modelscope
```

The downloader tries the remaining endpoints per file if one fails, verifies
`Content-Length`, and checks every `.safetensors` header. The checkpoint is
sharded; inspect it without loading all weights:

```python
from machaon.adapters import Checkpoint

ckpt = Checkpoint("./qwen36-27b")
print(ckpt.summary())
print(ckpt.find(layer=60, suffix="down_proj.weight"))
```

## 3 · Choose the memory working point

| mode | approximate weights | use |
|---|---:|---|
| bf16 | ~54 GB | larger-memory host |
| int8 | ~27 GB | tight on a 34 GB CPU/MPS host |
| int4 | ~14 GB | preferred for local evaluation |
| checkpoint-only write | one shard (~4 GB) | save an already-solved delta |

A 34 GB unified-memory laptop should use **int4 plus a local dense window**, or
perform the write with `Checkpoint.write_tensor` after obtaining the delta.

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
```

The edited layer and its successors may be densified for a gradient solve:

```python
runtime.densify_from(60)     # keep layers 0–59 quantized
```

Start near the output (`56`–`60` of 64). Moving the edit earlier increases both
the solve cost and the downstream perturbation.

## 4 · Solve a constrained edit

The target, protected prompts, budget, and solver are explicit:

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
    prompt="The Eiffel Tower is located in the city of",
    target=" Rome",
    layer=60,
    preserve_prompts=neutral,
    solver="gradient",
    steps=40,
    lr=0.02,
    max_rel_norm=0.005,
)
print(proposal.module)
print(proposal.relative_delta)
print(proposal.target_probability)
```

For a quick first pass, `solver="direction"` avoids the backward graph. For a
smaller, cheaper delta, use `solver="gradient"` with a local dense window.

## 5 · Verify and write one shard

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

`write_tensor` rewrites the shard containing the edited tensor and hard-links
the untouched shards. The source checkpoint remains unchanged. This is the
recommended save path for a 27B model: the peak write footprint is one shard,
not a second full checkpoint.

## 6 · Gate the result

Use a session to keep the baseline, proposal, apply, and rollback in one place:

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
session.rollback()       # or keep the edit after the gate accepts it
```

For a full `EditSuite`/`judge` run, use the same before/after model with a
checkpoint snapshot rather than loading two 27B copies. The gates are
fail-closed: a missing measurement rejects the edit. Start with a small budget
(`0.003`–`0.01`) and inspect the protected-key drift before widening it.

## 7 · Drive the same workflow from the browser

The local console uses the same session and the same checkpoint allowlist:

```bash
pip install -e ".[web]"
machaon-web --allowed-root ./qwen36-27b --port 8765
```

Open the printed URL, inspect the checkpoint header, then open the session with
`dtype=auto`, `quantize=int4`, `device=cpu`, layer `60` and budget `0.01`.
Capture the neutral prompts as the baseline, propose the edit, and read the
relative delta before pressing **Apply**. The evidence column streams job
state and applies the fail-closed gate after evaluation. A rollback restores
the original tensor bytes without reloading the checkpoint.

## Notes

- The vision encoder is not the edit target. A text-only prompt exercises the
  language decoder; image/video inputs remain available through the same model.
- `Qwen/Qwen3.6-27B` is a multimodal wrapper with a 64-layer text decoder;
  `HFModel` discovers `model.language_model.layers` before the text-only
  `model.layers` path.
- The model is licensed under Apache-2.0; checkpoint and dataset licences remain
  with their upstream sources.
