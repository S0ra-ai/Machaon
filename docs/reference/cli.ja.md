# CLI

```bash
python -m machaon <command>
```

終了コードは契約の一部で、CI ステップとして直接使えます：

| 終了コード | 意味 |
|---|---|
| 0 | 成功 / gate 通過 |
| 1 | gate が編集を拒否、または検証失敗 |
| 2 | 入力形式エラー、ファイル欠損、未対応チェックポイント |

## `fetch`

torch を読み込まず、重み tensor も展開せずに safetensors チェックポイントを
ダウンロードします。Hugging Face、`hf-mirror`、ModelScope の順にリポジトリを
一覧し、各 `.safetensors` のヘッダーを検証してから受け入れます。既存ファイルは
再利用するため、中断したダウンロードを再開できます。

```bash
machaon fetch --list
machaon fetch --starter qwen05b --dest ./models/Qwen2.5-0.5B
machaon fetch --starter smol135m --dest ./models/SmolLM2-135M
machaon fetch --repo Qwen/Qwen2.5-0.5B --dest ./models/qwen
```

`--endpoint huggingface|hf-mirror|modelscope` で 1 つの端点を強制できます。
`--json` はファイル一覧、シャード数、tensor 数、パラメータ数を出力します。
`.safetensors` 重みがないリポジトリは、中途半端なディレクトリを残さず拒否します。

## `start`

最初の編集までの最短経路と、インストール済みのオプション実行環境を表示します。

```bash
python -m machaon.cli start
python -m machaon.cli start --demo
```

## `intent`

自然言語の教学リクエストを、モデルを読み込まずに正確な `prompt` / `answer`
契約へコンパイルします。終了コード 2 は 1 組に確定できなかったことを示し、
エラーに受け付ける形式が表示されます。

```bash
python -m machaon.cli intent "エッフェル塔は -> ローマ"
python -m machaon.cli intent "教会模型：北京的首都是 → 北京" --json
```

矢印とラベル形式は正確です。普通の文も受け付けますが、分割点が仮定なので
`medium` 信頼度として表示されます。等価な `machaon teach` コマンドも出力します。

## `teach`

ベースライン、深さ方向の層検索、適用、目標/panel/KL チェック、失敗時の
ロールバックまでを 1 コマンドで実行します。

```bash
python -m machaon.cli teach --model ./models/Qwen2.5-0.5B \
  --say "エッフェル塔は -> ローマ"
```

既定の出力は `<checkpoint>-machaon` という兄弟ディレクトリで、元の
チェックポイントは変更しません。主なオプション:

| フラグ | 意味 |
|---|---|
| `--say TEXT` | `machaon intent` で文をコンパイル |
| `--prompt P --answer A` | 契約を明示 |
| `--layers 8,10,12` / `--layers 8-12` | 候補層を指定 |
| `--max-layers 7` | 既定の深さウィンドウの候補数 |
| `--solver direction\|gradient` | 各候補の value solver |
| `--max-rel-norm 0.02` | 1 編集あたりの相対ノルム予算 |
| `--budget 0.10` | セッション累積予算 |
| `--max-kl 0.05` | KL p95 の受け入れ閾値 |
| `--min-panel-agreement 0.875` | 無関係 prompt の top-1 保持率の下限 |
| `--preserve-file FILE` | 1 行 1 prompt のベースライン/投影集合 |
| `--corpus-file FILE` | 任意の perplexity チェック用テキスト |
| `--out DIR` | 編集済みチェックポイント（既存なら拒否） |
| `--no-save` | メモリ内だけに保持 |
| `--dry-run` | 検索とレポートのみ、書き込みなし |
| `--evidence-out DIR` | 未編集 headroom 付きの WSP 証拠を書き出す |
| `--edit-out DIR` | 適用済み delta を OpenWSP v1 として書き出す |
| `--report-out FILE` | 完全なレシピレポート JSON を書き出す |
| `--json` | 機械可読出力 |

終了コード: `0` 適用 / 変更不要 / ドライラン、`1` 拒否してロールバック、
`2` 入力またはモデルエラー。

`--out`、`--evidence-out`、`--edit-out` の出力先が既に存在する場合、以前の成果物を上書きしないよう拒否します。

## `teach-many`

複数の事実を 1 つのチェックポイント連鎖として教えます。各事実は、直前の
受け入れ済みチェックポイントから読み込んだ新しい session 上で単一の事実の
レシピ全体を実行します。元のチェックポイントは変更しません。

```bash
machaon teach-many --model ./models/Qwen2.5-0.5B \
  --facts facts.txt --out ./models/qwen-many
```

`facts.txt` は 1 行 1 文（`machaon intent` と同じ文法）か、JSON / JSONL の
`{"say": ...}` または `{"prompt": ..., "answer": ...}` のリストです。最後の
全体として安全なチェックポイントを `<out>/final` に置き、各ステップの証拠を
`<out>/report.json` に書きます。事実が拒否された場合や累積ドリフト検査が
失敗した場合は終了コード 1 です。`--continue-on-reject` は失敗を記録して
続行し、`--keep-steps` は最後の安全なステップだけでなく全ステップを保持します。
この場合の final は `<out>/final` ではなく最後の `step-NNN` ディレクトリです。
累積 `--budget` は事実ごとではなく連鎖全体に適用されます。

## `demo`

合成 rank-one デモを実行。モデルもネットワークも不要。

```bash
python -m machaon.cli demo
```

## `validate`

edit spec を検証。spec が無効なら終了コード 1、JSON が壊れていれば 2。

```bash
python -m machaon.cli validate examples/edit_spec.json
# OK  rank1_constrained -> model.layers.14.mlp.down_proj (budget 0.020)
```

## `gate`

metrics JSON にゲートを適用。終了コード 1 はコミットすべきでないことを意味し、CI に直接入れられます。

```bash
python -m machaon.cli gate examples/metrics_pass.json    # exit 0
python -m machaon.cli gate examples/metrics_fail.json    # exit 1
```

```
FAIL
  paraphrase_acc = 0.41 violates >= 0.6
  kl_p95 = 0.113 violates <= 0.05
  ...
```

## `inspect`

safetensors チェックポイントの tensor を一覧。

```bash
python -m machaon.cli inspect /models/qwen --suffix down_proj.weight
python -m machaon.cli inspect /models/qwen --layer 3 --limit 5
python -m machaon.cli inspect /models/qwen --layer 3 --json
```

## `doctor`

インストール済みの能力とバージョンを報告。issue に貼るのに向いています。

```bash
python -m machaon.cli doctor
python -m machaon.cli doctor --json
```

## `--version`

```bash
python -m machaon --version
```

## CI

```bash
./scripts/ci.sh
```

テスト、デモ、spec 検証、2 つのゲートケース、lint を実行します。
