# インストール

## 要件

| | |
|---|---|
| Python | 3.9、3.10、3.11、3.12、3.13 |
| コア | NumPy のみ |
| 実モデル | PyTorch + transformers + safetensors |
| ベンチマーク | pyarrow |
| MCP サーバー | `mcp` SDK、Python 3.10+ |
| ブラウザーコンソール | FastAPI + uvicorn |

## インストール

```bash
git clone https://github.com/S0ra-ai/Machaon.git
cd Machaon
uv venv && uv pip install -e ".[dev]"
```

オプション：

```bash
uv pip install -e ".[hf]"       # torch + transformers + safetensors
uv pip install -e ".[bench]"    # pyarrow、MMLU / WikiText ローダー
uv pip install -e ".[mcp]"      # MCP SDK、Python 3.10+
uv pip install -e ".[web]"      # ローカルブラウザーコンソール
uv pip install -e ".[dev,docs]" # テスト、lint、カバレッジ、ドキュメント
```

## 検証

```bash
./scripts/ci.sh
```

テスト、合成デモ、spec 検証、2 つのゲートケース、lint を実行します。ネットワークも GPU も不要です。

想定される末尾：

```
Ran 590 tests ...
ALL CHECKS PASSED
```

## モデルのダウンロード

`machaon fetch` は重みを読み込まずに safetensors チェックポイントを取得します。
Hugging Face、`hf-mirror`、ModelScope の順に試し、各 safetensors ヘッダーを
検証して、既存ファイルを再利用します。

```bash
machaon fetch --list
machaon fetch --starter qwen05b --dest ./models/Qwen2.5-0.5B
machaon fetch --repo Qwen/Qwen2.5-1.5B --dest ./qwen15b
```

`HF_ENDPOINT` は Transformers ローダーには引き続き効きますが、組み込み
ダウンローダーは独自のフォールバックを使います。`.safetensors` 重みがない
リポジトリは、中途半端なディレクトリを残さず拒否します。

ベンチマークデータは別途取得します：

```bash
python scripts/fetch_benchmarks.py
```

ファイルは `$MACHAON_BENCH`（既定 `~/.cache/machaon/bench`）に置かれ、それぞれ `.provenance.json` にソース、revision、ライセンス、件数、SHA-256 を記録します。

## トラブルシューティング

**macOS で `RuntimeWarning: divide by zero encountered in matmul`** —— Accelerate の浮動小数点フラグの癖です。Machaon 自身のカーネル内では抑制しています。自分のコードで出る場合はプラットフォームの挙動で、数学の誤りではありません。

**`this runtime has no tokenizer`** —— ID ベースの runtime でテキストメソッドを呼んでいます。token id を渡すか、`HFModel(model, tokenizer)` を構築してください。
