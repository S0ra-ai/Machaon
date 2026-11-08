# Qwen3.6-27B の重み空間プログラミング

Qwen3.6-27B は視覚エンコーダ付きのマルチモーダルチェックポイントです。
テキストデコーダは 64 層で、MLP 出力投影は次にあります。

```text
model.language_model.layers.<layer>.mlp.down_proj.weight
```

Machaon はこのマルチモーダル経路を先に検出し、通常の `model.layers` に
フォールバックします。

## 1 · 対応 runtime のインストール

Qwen3.6 には新しい Transformers と `qwen3_5` 対応が必要です。独立環境を推奨します。

```bash
uv venv .venv-qwen36
source .venv-qwen36/bin/activate
uv pip install -e ".[hf]"
uv pip install -U "transformers[serving]" torchvision pillow optimum-quanto
```

`machaon.runtime.load_pretrained` は config に応じて
`AutoModelForImageTextToText` または `AutoModelForCausalLM` を選びます。

## 2 · ダウンロードと検査

```bash
python scripts/fetch_model.py \
  --repo Qwen/Qwen3.6-27B \
  --dest ./qwen36-27b \
  --endpoint modelscope
```

全重みをロードせず header だけ確認します。

```python
from machaon.adapters import Checkpoint

ckpt = Checkpoint("./qwen36-27b")
print(ckpt.summary())
print(ckpt.find(layer=60, suffix="down_proj.weight"))
```

## 3 · メモリ上の作業点

| モード | 重み | 用途 |
|---|---:|---|
| bf16 | 約 54 GB | 大メモリ/GPU |
| int4 | 約 14 GB | 34 GB マシンでの評価 |
| checkpoint-only | 1 shard（約 4 GB） | 既知 delta の書き込み |

34 GB では int4 と局所 dense window を使い、56–60 層付近を編集します。

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

## 4 · 制約付き編集を解く

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

最初の確認には `solver="direction"` も使えます。

## 5 · 検証と shard 書き込み

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

対象 tensor の shard だけを書き換え、残りはハードリンクします。元の
チェックポイントは変更されません。27B ではこの保存経路が実用的です。

## 6 · Session で管理する

```python
from machaon.session import Session

session = Session(runtime, "./qwen36-27b")
session.baseline(neutral, held_out_documents)
session.propose(
    "The Eiffel Tower is located in the city of", " Rome", layer=60,
    preserve_prompts=neutral, solver="gradient", steps=40, lr=0.02,
    max_rel_norm=0.005,
)
session.apply()
metrics = session.evaluate(
    ["The Eiffel Tower is located in the city of"],
    [tokenizer(" Rome", add_special_tokens=False)["input_ids"][0]],
)
session.rollback()
```

`EditSuite`/`judge` には同じモデルの before/after 状態または checkpoint
スナップショットを使い、27B を二重にロードしないでください。
