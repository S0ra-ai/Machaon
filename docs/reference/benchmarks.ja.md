# ベンチマーク

ここにあるすべての数値は、ライセンスと SHA-256 を記録した公開データセットから来ています。ファイルは `scripts/fetch_benchmarks.py` が取得し、隣に `.provenance.json` を書きます。

## データセット

| データセット | ソース | ライセンス | 件数 | SHA-256 |
|---|---|---|---|---|
| zsRE（development） | `hf://datasets/wangzn2001/zsre/structured_zeroshot-dev-new_annotated_final.jsonl` | ミラーは未宣言；上流 zsRE、Levy et al. 2017 | 10,813 | `05bbeb0ed8c4…` |
| WikiText-2 raw（test） | `hf://datasets/Salesforce/wikitext/wikitext-2-raw-v1/test` | CC-BY-SA-3.0 AND GFDL | 4,358 | `5f1bea067869…` |
| MMLU（test） | `hf://datasets/cais/mmlu/all/test` | MIT | 14,042 | `74a41822ce7d…` |

!!! warning "zsRE ライセンス"
    ここで使うミラーはライセンスを宣言していません。zsRE は知識編集文献（MEND、SERAC、MEMIT）で広く使われていますが、再配布する場合は元ソースで条件を確認してください。

いつでも検証：

```python
from machaon.bench import read_provenance, verify
verify("~/.cache/machaon/bench/zsre-dev.jsonl")
```

## プロトコル

**Target injection。** 各 zsRE 質問について、データセット自身の実体語彙から妨害 token を選び、モデルがそれを予測するよう編集し、元の質問と人手の言い換えを測ります。

なぜ「モデルが間違える項目の修復」ではないのか？Qwen2.5-0.5B は zsRE の最初の token を 300 個中 299 個正解するので、そのプロトコルには改善余地がほとんどありません。Target injection は弱いモデルにも強いモデルにも効きます。

**WikiText-2 — パープレキシティ。** 編集前後で留保文書の平均 NLL を取り、`exp(Δ mean NLL)` を報告。

**MMLU — 能力。** 四択問題を文字補完の平均 log-probability で採点。

**ソルバー。** `gradient` 200 ステップ、モデルパラメータは凍結し、backward は value ベクトルの勾配だけを蓄積。CPU で得た 2 つの発見：

| 観察 | 測定 |
|---|---|
| macOS CPU の bf16 backward | 4.66 秒/step |
| macOS CPU の float32 backward | **0.18 秒/step**（26 倍速い） |
| Qwen2.5-0.5B 層 12、1 編集 | rel. delta 0.150 |
| 同モデル層 20 | rel. delta **0.0043**（35 倍安い） |

どちらもソルバー超参より重要です。CPU では float32 を使い、振る舞いを変えられる範囲で出力に最も近い層を選びます。

## 結果

同じプロトコル、同じ 8 編集、同じ評価器。

| モデル | パラメータ | 層 | 平均 rel.Δ | rewrite | rephrase（513） | WikiText-2 | MMLU |
|---|---|---|---|---|---|---|---|
| GPT-2 124M | 124M | 8 / 12 | 0.0135 | 0.500 | 0.259 | 1.066 | 0.220 → 0.220（n=100） |
| Qwen2.5-0.5B | 494M | 20 / 24 | 0.0133 | 0.375 | 0.275 | 1.030 | 0.440 → 0.440（n=50） |
| Qwen2.5-1.5B | 1544M | 20 / 28 | **0.0065** | 0.500 | 0.273 | **1.005** | 0.400 → 0.400（n=50） |
| Qwen2.5-3B | 3086M | 34 / 36 | 0.0072 | 0.625 | **0.327** | 1.009 | 0.500 → 0.500（n=30） |

1.5B は 100 ステップ、3B は層 34 で 30 ステップ。傾向が重要です：0.5B から 3B へ、同じ injection の相対 delta は約半分、パープレキシティコストは 3.0% から 0.9% に下がります。3B チェックポイントはシャード（2 ファイル）で、`AutoModelForCausalLM` 経由で各層の `down_proj` を 1 つ編集します。

### 1.5B のファイルレベル書き込み

```
source   : 338 tensors, 1543.7M params, 1 shard(s)
target   : model.layers.20.mlp.down_proj.weight [1536, 8960]
delta    : rel 0.0050
wrote    : /tmp/qwen15b_edited3 (2.3s)
untouched: model.embed_tokens.weight identical=True
```

2.9 GB のチェックポイントを 2.3 秒で書き換え、1 tensor だけ変更、他はバイト一致。7–8B 向けのパスで、モデルと 2 部目を同時にメモリへ置けない場合に使います。

## 32 GB ノート PC 上の 7B

Qwen2.5-7B は 7.6B パラメータ、4 シャードのチェックポイント。**int8 重み**で 32 GB マシン上に載り、常駐約 9.4 GB、forward 2.5 秒。

**ファイルレベル書き込み**は 1 シャードを書き換え、残りをハードリンク：

```
source   : 339 tensors, 7615.6M params, 4 shard(s)
target   : model.layers.24.mlp.down_proj.weight [3584, 18944]
delta    : rel 0.0030
shard    : model-00004-of-00004.safetensors (linked 3, copied 0)
wrote    : /tmp/qwen7b_edit (2.3s)
untouched: lm_head.weight identical=True
```

ピークメモリは 1 シャード（約 4 GB）で、チェックポイント全体ではありません。

**フル評価**は同じ int8 モデルで動きます：`optimum-quanto` が約 9.4 GB に保ち、`HFModel.densify_from(layer)` が編集層以降を float32 にするので、backward は 28 ブロック中 2 つだけを通ります。

| solver | 平均 rel.Δ | rewrite | rephrase | WikiText-2 | ピーク RSS |
|---|---|---|---|---|---|
| `direction`（backward なし） | 0.0503 | 0.750 | 0.667 | 1.260 | 10.6 GB |
| **`gradient`、ローカル** | **0.0043** | **1.000** | **0.929** | **1.007** | 10.2 GB |

設定：28 層中 26 層、30 ステップ、lr 0.05、int8 重みで `lm_head` は量子化除外、3 編集、各 15 言い換え。

### 出力までの深さ

7B 設定で 3B（36 層中 26 層、残り 10 ブロック）を再実行：

| 3B 設定 | 後続ブロック | 平均 rel.Δ | rewrite | rephrase | WikiText-2 |
|---|---|---|---|---|---|
| 層 26、lr 0.02、80 ステップ | 10 | 0.0052 | 0.625 | 0.230 | 1.002 |
| 層 26、lr 0.05、30 ステップ | 10 | 0.0079 | 0.500 | 0.236 | 1.002 |
| 層 32、lr 0.05、30 ステップ | 4 | 0.0068 | 0.625 | 0.312 | 1.009 |
| **層 34、lr 0.05、30 ステップ** | **2** | 0.0072 | 0.625 | **0.327** | 1.009 |

深さは汎化を助け（0.230 → 0.327）、8 ターゲットで頭打ちになります。

**同ターゲット数の実行が読み方を修正します。** 7B 実験と同じ 3 ターゲット、42 言い換えで、3B はほぼ同じ数値に達します：

| モデル | 編集数 | 平均 rel.Δ | rewrite | rephrase | WikiText-2 |
|---|---|---|---|---|---|
| Qwen2.5-3B、層 34 | 3 | 0.0082 | **1.000** | **0.857**（36/42） | 1.0068 |
| Qwen2.5-7B、層 26（int8） | 3 | 0.0043 | **1.000** | **0.929**（39/42） | 1.0070 |

早い 0.625 は 8 ターゲットサンプルの性質（最後の 5 項目が難しい）で、硬い容量上限ではありません。同じサンプル数では、主な差はコストです：3B は同じ結果に約 2 倍の相対 delta を要します。

## 8B：現在の最大実測モデル

Llama-3.1-8B は 8,030.3M パラメータ、4 シャードで、宣言した上限内。32 GB マシンで 2 構成：

| 構成 | 平均 rel.Δ | rewrite | rephrase | WikiText-2 | ピーク RSS |
|---|---|---|---|---|---|
| int8、層 31/32（1 ブロック密） | 0.0072 | **1.000** | 0.500 | 1.284 | 10.2 GB |
| int4、層 30/32（2 ブロック密） | 0.0138 | 0.500 | 0.500 | **1.031** | 8.6 GB |

各 2 編集、20 言い換え、WikiText 2 文書、MMLU n=8。

**int8** は 1 つの密ブロックで 10.2 GB、rewrite 1.000。**int4** は 2 つの密ブロックで 8.6 GB、パープレキシティ 1.031。

**7.6B int8 + 2 ブロックウィンドウ**が rewrite 1.000 / rephrase 0.929 / パープレキシティ 1.007 で、現在の最強組み合わせです。

`--quantize int4` は quanto の unpack kernel 用に `ninja` が PATH に必要です。

## なぜ微調整しないのか

`examples/lora_baseline.py` は同じ層、同じ 3 prompt、同じターゲット token、同じ 30 勾配ステップで rank-8 adapter を訓練します。Qwen2.5-0.5B、float32：

| 手法 | rewrite | rephrase | WikiText-2 | MMLU（n=8） |
|---|---|---|---|---|
| rank-1 閉形式、30 ステップ | 1.000 | 0.976（41/42） | **1.044** | 0.500 → 0.625 |
| LoRA rank 8、30 ステップ | 1.000 | **1.000** | **1.441** | 0.500 → **0.250** |

LoRA は表現力が高く、留保言い換えを 1.000 にします。一方 46,080 の無制約パラメータを使い、モデルを 44% のパープレキシティと 25 点の MMLU 低下に連れていきます。rank-one 編集は保護 key に投影され予算で制限されるため、10 分の 1 の損傷でほぼ同じ有効性に達します。

## 累積コスト

`examples/budget_sweep.py` は 1 編集ずつ適用し、チェックポイントで留保パープレキシティを測ります。GPT-2 124M、層 8、100 ステップ：

| 編集数 | 累積 Δ | WikiText-2 パープレキシティ |
|---|---|---|
| 0 | 0.000 | 1.000 |
| 8 | 0.089 | 1.049 |
| 16 | 0.151 | 1.098 |
| 32 | 0.255 | 1.191 |

`EditBudget(max_total=0.10)` は 10 回目で拒否。累積相対変化が約 0.1 を超えると 124M–0.5B モデルは目に見えて劣化する、というのが既定予算の経験的根拠です。

## 長文一貫性

`examples/longform_check.py` は Eiffel Tower 編集を適用し、5 prompt の 120 token greedy 生成を比較します。一貫性は**未編集**モデルで採点。

| モデル | baseline 反復 | 編集後反復 | Δ | NLL ratio | 空 |
|---|---|---|---|---|---|
| GPT-2 124M（120 tok） | 0.621 | 0.578 | **−0.043** | 1.089 | 0.0 |
| Qwen2.5-1.5B（80 tok） | 0.097 | 0.196 | **+0.099** | 0.938 | 0.0 |

GPT-2 では反復がほぼ変わらず、1.5B では約 2 倍になり 0.10 ゲートにちょうど当たります。これは警告で、祝うべき合格ではありません：しきい値が緩い可能性があり、80 token は短い。

## 再現

```bash
export HF_ENDPOINT=https://hf-mirror.com
python scripts/fetch_benchmarks.py

HF_HUB_OFFLINE=1 python examples/bench_edits.py \
  --model gpt2 --layer 8 --edits 8 --dtype float32 --solver gradient

HF_HUB_OFFLINE=1 python examples/bench_edits.py \
  --model /path/to/Qwen2.5-0.5B --layer 20 --edits 8 \
  --dtype float32 --solver gradient --mmlu 50 --wikitext 20
```

## 次の測定

1. 各構成 100+ 編集と分散推定。
2. モデル横断の累積予算スイープ。
3. 各規模の長文一貫性。
4. 3B と 7B の LoRA baseline。
