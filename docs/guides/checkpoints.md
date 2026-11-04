# Save a checkpoint

`machaon.adapters.Checkpoint` reads and rewrites safetensors without ever
touching the source.

## Open

```python
from machaon.adapters import Checkpoint

ckpt = Checkpoint("/models/qwen")            # directory with an index, or a file
ckpt.names()[:3]
ckpt.info("model.layers.3.mlp.down_proj.weight")
ckpt.find(layer=3, suffix="down_proj.weight")
```

Supported layouts:

- a single `.safetensors` file
- a directory containing exactly one
- a sharded checkpoint with `model.safetensors.index.json`

## Read

```python
W = ckpt.read(name)                # float64, exact widening from bf16/fp16
W = ckpt.read(name, dtype=None)    # keep the stored dtype
```

Reads are lazy: one tensor at a time, so a 7B sharded checkpoint does not need to
fit in memory all at once.

## Write

```python
report = ckpt.write_with({name: delta}, "/models/edited")
```

- the source is never modified;
- untouched tensors are copied bit-for-bit;
- edited tensors are added in float64 and cast back to their original dtype;
- sharded checkpoints keep their shard layout and index.

`report["tensors"][name]` gives the relative change and the number of changed
entries.

## Refuse bad writes

```python
from machaon.refusals import check_tensor

check_tensor(name, value, original,
             frozen={"model.embed_tokens.weight", "lm_head.weight"})
```

Rules: frozen tensors, shape mismatch, non-finite values, bfloat16 overflow, and
whole-tensor scalar rescaling (which is how you quietly shrink a model).
