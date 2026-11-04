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

## Inspect without loading

```python
ckpt.summary()
# {'tensors': 339, 'parameters': 1543714304, 'dtypes': {'BF16': ...}, 'shards': 1, ...}
```

Reads headers only. Useful before you commit to loading a multi-gigabyte model.

## Download a model

```bash
python scripts/fetch_model.py --repo Qwen/Qwen2.5-1.5B --dest ./qwen15b
```

Tries `huggingface.co`, then `hf-mirror.com`, then ModelScope, and verifies every
`.safetensors` header rather than trusting the transfer.

## Read

```python
W = ckpt.read(name)                # float64, exact widening from bf16/fp16
W = ckpt.read(name, dtype=None)    # keep the stored dtype
```

Reads are lazy: one tensor at a time, so a 7B sharded checkpoint does not need to
fit in memory all at once.

## Write one tensor, one shard

For checkpoints too large to hold twice, rewrite only the shard that contains
the tensor and hard-link the rest:

```python
report = ckpt.write_tensor(name, W + delta, "/models/edited")
# {'shard': 'model-00004-of-00004.safetensors', 'linked': [...3 files...], 'copied': []}
```

Peak memory is one shard. On a 7.6B model that is ~4 GB instead of 14 GB, and
the write took 2.3 s.

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
