# クイックスタート

5 分、ダウンロードなし。ここにあるコマンドはすべてオフラインで動きます。

## 1 · デモを実行

```bash
python -m machaon.cli demo
```

```
op                         rel.delta   target err   preserve drift probe drift p95
----------------------------------------------------------------------------------
rank1 (naive)                 0.0509     1.60e-16         8.71e-02          0.0902
constrained (no cap)          0.0730     2.02e-16         3.82e-16          0.1394
constrained (capped)          0.0200     2.84e-01         2.52e-16          0.0382
```

3 列目を見てください。素朴な編集は保護 key を 8.7e-2 動かし、制約付き編集は 3.8e-16 に留めます。

## 2 · 自分の行列を編集

```python
import numpy as np
from machaon import rank1_constrained
from machaon.verify import check_protected

rng = np.random.default_rng(0)
W = rng.normal(0, 0.5, size=(64, 48))      # (d_out, d_in)
keys = rng.normal(size=(16, 48))           # 動いてはいけない
k = rng.normal(size=48)                    # 記憶する
v = W @ k + rng.normal(0, 1.0, size=64)    # 出力する

W_new, delta = rank1_constrained(W, k, v, preserve=keys, max_rel_norm=0.05)

print(check_protected(W, delta, keys))
```

```
{'passed': True, 'max_drift': 3.1e-16, ...}
```

## 3 · トランザクションに入れる

```python
from machaon import Workspace

ws = Workspace()
ws.register("layer.8.mlp.c_proj", W)

txn = ws.begin("first edit")
txn.apply("layer.8.mlp.c_proj", delta)

# ブール値の `.passed` を持つものなら何でも verdict になれる
txn.commit()          # または txn.rollback()
```

`commit(verdict)` は `verdict.passed` が false のとき拒否し、元のバイトを復元します。

## 4 · 実モデルで実行

```bash
export HF_ENDPOINT=https://hf-mirror.com
python examples/gpt2_rome.py
```

```
model      : gpt2 (12 layers)
edit       : transformer.h.8.mlp.c_proj.weight
delta      : rel 0.0153
  rewrite_acc  1
  paraphrase_acc 1
  kl_p95       0.000614724
  ppl_ratio    1.00079
PASS (5 gates)
```

## 次へ

- [事実を編集する](../guides/edit-a-fact.md) —— 手順を一つずつ。
- [編集を評価する](../guides/evaluate.md) —— ゲート、KL、ベンチマーク。
- [数学的原理](../MATH.md) —— 導出と失敗モード。
