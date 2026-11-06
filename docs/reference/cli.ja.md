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
