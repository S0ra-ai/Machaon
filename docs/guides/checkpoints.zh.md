# 保存检查点

`machaon.adapters.Checkpoint` 读取和重写 safetensors，源文件始终不被修改。

## 打开

```python
from machaon.adapters import Checkpoint

ckpt = Checkpoint("/models/qwen")            # 带 index 的目录，或单个文件
ckpt.names()[:3]
ckpt.info("model.layers.3.mlp.down_proj.weight")
ckpt.find(layer=3, suffix="down_proj.weight")
```

支持的布局：

- 单个 `.safetensors` 文件
- 恰好包含一个 safetensors 文件的目录
- 带 `model.safetensors.index.json` 的分片检查点

## 不加载也能检查

```python
ckpt.summary()
# {'tensors': 339, 'parameters': 1543714304, 'dtypes': {'BF16': ...}, 'shards': 1, ...}
```

只读 header。在决定加载多 GB 模型之前很有用。

## 下载模型

```bash
python scripts/fetch_model.py --repo Qwen/Qwen2.5-1.5B --dest ./qwen15b
```

依次尝试 `huggingface.co`、`hf-mirror.com`、ModelScope，并校验每个 `.safetensors` header，而不是盲信传输。

## 读取

```python
W = ckpt.read(name)                # float64，从 bf16/fp16 精确加宽
W = ckpt.read(name, dtype=None)    # 保持存储 dtype
```

读取是惰性的：一次一个 tensor，因此 7B 分片检查点可以流式处理，不需要一次性放进内存。

## 写一个 tensor，一个分片

对于大到无法同时持有两份的检查点，只重写包含该 tensor 的分片，其余硬链接：

```python
report = ckpt.write_tensor(name, W + delta, "/models/edited")
# {'shard': 'model-00004-of-00004.safetensors', 'linked': [...3 files...], 'copied': []}
```

峰值内存是一个分片。在 7.6B 模型上约 4 GB，而不是 14 GB；写入耗时 2.3 秒。

## 写入

```python
report = ckpt.write_with({name: delta}, "/models/edited")
```

- 源文件从不修改；
- 未触碰的 tensor 逐位复制；
- 编辑过的 tensor 用 float64 相加，再转回原 dtype；
- 分片检查点保持原有分片布局和 index。

`report["tensors"][name]` 给出相对变化和改动条目数。

## 拒绝坏写入

```python
from machaon.refusals import check_tensor

check_tensor(name, value, original,
             frozen={"model.embed_tokens.weight", "lm_head.weight"})
```

规则：冻结 tensor、形状不匹配、非有限值、bfloat16 溢出，以及整 tensor 标量缩放（悄悄缩小模型的常见手法）。
