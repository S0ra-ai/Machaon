# クイックスタート

5 分。最初の 2 ステップはオフラインで動きます。教学ステップはローカルの
チェックポイントを使います。

## 0 · ゼロから 1 コマンド

```bash
machaon first
```

新しいマシンでは SmolLM2-135M（約 270 MB）を取得し、開いて
"The Eiffel Tower is in the city of -> Rome" を教え、受け入れられた編集を
`./models/SmolLM2-135M-machaon` に保存します。`--say` で事実を変更し、
`--model` で既存チェックポイントを指定し、`--no-save` でメモリ内だけに保持できます。

## 1 · 指定したチェックポイントを教える

チェックポイントから検証済みの編集まで、最短経路は 1 コマンドです。

0.5B チェックポイントを一度だけ取得します（約 1 GB、ミラーも自動で試します）：

```bash
machaon fetch --starter qwen05b --dest ./models/Qwen2.5-0.5B
machaon fetch --list          # smol135m / qwen05b / gpt2
```

ダウンローダーは標準ライブラリのみを使い、各 safetensors ヘッダーを検証し、
既存ファイルを再利用します。

```bash
machaon start
machaon intent "エッフェル塔は -> ローマ"

machaon teach --model ./models/Qwen2.5-0.5B \
  --say "エッフェル塔は -> ローマ"
```

`intent` はモデルを読み込まず、文を正確な `prompt` と `answer` にコンパイル
します。`teach` は続いて:

1. モデルがすでに答えられるかを確認し;
2. 無関係なベースライン panel を自動で取得し、その零空間へ編集を投影し;
3. 深さ方向の候補層を順に適用・評価・ロールバックし;
4. 最良の層を再求解して予算内で適用し;
5. 目標 token、panel の top-1 保持率、KL を測定し;
6. 必須チェックがすべて通ったときだけ編集を残します。失敗時は元のバイトを
   復元します。

既定の出力は `<checkpoint>-machaon` という兄弟ディレクトリで、元の
チェックポイントは変更しません。`--dry-run` は書き込まずに検索し、
`--no-save` はメモリ内だけに留め、層が分かっている場合は
`--layers 8,10,12` を使います。

対応する文の形:

```text
エッフェル塔は -> ローマ
prompt: エッフェル塔は
answer:  ローマ
when asked "the Eiffel Tower is", answer " in Paris"
教会模型：北京的首都是 → 北京
```

矢印とラベル形式は正確です。普通の文も受け付けますが、分割点が仮定なので
中程度の信頼度として表示されます。ローカルチェックは目標と宣言した panel
だけを対象にします。ベンチマーク水準の主張には完全な受け入れレポートが必要です。

教学が固定するのは答えの**最初の token** です。複数 token の答えは警告として
表示され、続きはベンチマークの問いであって、このローカルチェックの保証では
ありません。
### 複数の事実を教える

```bash
cat > facts.txt <<'EOF'
エッフェル塔は -> ローマ
フランスの首都は -> パリ
EOF

machaon teach-many --model ./models/Qwen2.5-0.5B --facts facts.txt --out ./models/qwen-many
```

各事実は独自の層検索とローカルチェックを持ちます。連鎖は最後の全体安全な
チェックポイントを `<out>/final` に置き、各ステップを `<out>/report.json` に
記録します。事実が拒否された場合は停止し、以前の事実を保持します。
`--continue-on-reject` は失敗を飛ばし、`--keep-steps` は連鎖全体を保持します。


行列、key、保護集合を自分で選びたい場合は、以下の手動ステップを使ってください。

## 2 · デモを実行

```bash
python -m machaon.cli demo
```

```
op                         rel.delta   target err   preserve drift probe drift p95
----------------------------------------------------------------------------------
rank1 (naive)                 0.0509     1.60e-16         8.71e-02          0.0902
constrained (no cap)          0.0730     2.02e-16         3.11e-16          0.1394
constrained (capped)          0.0200     2.84e-01         8.55e-17          0.0382
```

3 列目を見てください。素朴な編集は保護 key を 8.7e-2 動かし、制約付き編集は 3.8e-16 に留めます。

プログラムの全経路（生成、実行、保存、再読込、レポート検証）をオフラインで試すには:

```bash
python examples/tiny_program.py
```

ローカルに小さなランダム GPT-2 を構築し、何もダウンロードしません。これは配線の確認であり、
言語モデルの性能結果ではありません。

## 3 · 自分の行列を編集

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

## 4 · トランザクションに入れる

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

## 5 · 実モデルで実行

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
- [数学的原理](../MATH.md) —— 導出、診断、適用範囲。
