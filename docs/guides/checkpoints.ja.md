# チェックポイントを保存する

`machaon.adapters.Checkpoint` は safetensors を読み書きし、ソースには一切触れません。

## 開く

```python
from machaon.adapters import Checkpoint

ckpt = Checkpoint("/models/qwen")            # index 付きディレクトリ、またはファイル
ckpt.names()[:3]
ckpt.info("model.layers.3.mlp.down_proj.weight")
ckpt.find(layer=3, suffix="down_proj.weight")
```

対応レイアウト：

- 単一の `.safetensors` ファイル
- ちょうど 1 つを含むディレクトリ
- `model.safetensors.index.json` 付きのシャードチェックポイント

## ロードせずに検査

```python
ckpt.summary()
# {'tensors': 339, 'parameters': 1543714304, 'dtypes': {'BF16': ...}, 'shards': 1, ...}
```

ヘッダーだけを読みます。数 GB のモデルをロードする前に有用です。

## モデルをダウンロード

```bash
python scripts/fetch_model.py --repo Qwen/Qwen2.5-1.5B --dest ./qwen15b
```

`huggingface.co`、`hf-mirror.com`、ModelScope の順に試し、転送を盲信せず各 `.safetensors` ヘッダーを検証します。

## 読み取り

```python
W = ckpt.read(name)                # float64、bf16/fp16 から正確に拡張
W = ckpt.read(name, dtype=None)    # 保存 dtype を維持
```

読み取りは遅延型：1 tensor ずつなので、7B シャードチェックポイントをメモリに全部載せずにストリームできます。

## 1 tensor、1 シャードを書く

2 部持てない大きさのチェックポイントでは、tensor を含むシャードだけ書き換え、残りをハードリンクします：

```python
report = ckpt.write_tensor(name, W + delta, "/models/edited")
# {'shard': 'model-00004-of-00004.safetensors', 'linked': [...3 files...], 'copied': []}
```

ピークメモリは 1 シャード。7.6B モデルで 14 GB ではなく約 4 GB、書き込みは 2.3 秒。

## 書き込み

```python
report = ckpt.write_with({name: delta}, "/models/edited")
```

- ソースは変更されない；
- 触れていない tensor はビット単位でコピー；
- 編集した tensor は float64 で加算し、元 dtype に戻す；
- シャードチェックポイントはレイアウトと index を維持。

`report["tensors"][name]` に相対変化と変更エントリ数が入ります。

## 悪い書き込みを拒否

```python
from machaon.refusals import check_tensor

check_tensor(name, value, original,
             frozen={"model.embed_tokens.weight", "lm_head.weight"})
```

ルール：凍結 tensor、形状不一致、非有限値、bfloat16 オーバーフロー、tensor 全体のスカラー再スケール（モデルを静かに縮める方法）。
