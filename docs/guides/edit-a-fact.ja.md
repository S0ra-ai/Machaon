# 事実を編集する

`machaon.procedures.rome` の実装に沿った完全な手順です。

## 6 つのステップ

### 1 · 行列を選ぶ

事実は通常、中間層の MLP ブロックにあります。深さの 60–70% から始め、能力が損なわれるなら浅い層へ移します。

```python
from machaon.runtime import HFModel

runtime = HFModel(model, tokenizer)
name = runtime.edit_target(layer=8)          # transformer.h.8.mlp.c_proj.weight
```

`edit_target` はモデル族間で名前を翻訳し（`down_proj` ⇄ `c_proj`）、重みの向きを正規化するので、数学層は常に `(d_out, d_in)` を見ます。

### 2 · key を取得

key は行列が**実際に**見る入力で、最終 layer norm 前の residual stream ではありません。

```python
ids = runtime.encode(["The Eiffel Tower is located in the city of"])[0]
key = runtime.capture_module_input("transformer.h.8.mlp.c_proj", ids)[0]
```

事実が 1 つのテンプレートに縛られないなら、複数の言い換えで平均します。

### 3 · 保護集合を作る

```python
neutral = ["The capital of Germany is", "Water freezes at a temperature of", ...]
preserve = np.stack([runtime.capture_module_input(module, runtime.encode([p])[0])[0]
                     for p in neutral])
```

動いてはいけない key です。8〜30 個で通常十分；保護を増やすほど有効性のコストが上がります。

### 4 · value を解く

value は行列がその key に対して出すべき出力です。ROME は短い勾配で最適化します。対象は**単一ベクトル**だけで、モデルではありません：

```python
result = rome_edit(runtime, prompt, target, layer=8,
                   preserve_prompts=neutral, steps=250)
```

### 5 · 予算付きで適用

```python
W_new, delta = rank1_constrained(W, key, value, preserve=preserve,
                                 max_rel_norm=0.02)
```

予算は `‖ΔW‖/‖W‖` の上限です。必要量が大きい場合、使いすぎずに少なめに届けます。

### 6 · 検証

```python
from machaon.verify import check_protected
check_protected(W, delta, preserve)     # max_drift は ~1e-16
```

## エンドツーエンド

```bash
export HF_ENDPOINT=https://hf-mirror.com
python examples/gpt2_rome.py
```

```
model      : gpt2 (12 layers)
edit       : transformer.h.8.mlp.c_proj.weight
solved p(t): 0.9968 after 250 steps
delta      : rel 0.0153
  rewrite_acc  1
  paraphrase_acc 1
  kl_p95       0.000614724
  ppl_ratio    1.00079
PASS (5 gates)
```

## 失敗モード

| 症状 | 原因 | 対処 |
|---|---|---|
| 正確な prompt は動くが言い換えが失敗 | key が 1 テンプレートに縛られている | 言い換えで平均 |
| 能力が落ちる | 層が中央すぎる | 2〜4 層浅くする |
| bf16 で何も変わらない | delta が丸めで消える | float64 で計算 |
| 無関係な prompt がドリフト | 保護集合がない | 追加する |
| "preserved subspace" で拒否 | target key が保護 span 内 | 集合を小さくする |
