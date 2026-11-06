# MCP サーバー

Machaon は Model Context Protocol 経由でファイルレベル操作を公開します。agent はモデルをロードせずにチェックポイントを検査し、他人の編集を監査し、delta を適用し、結果をゲートできます。

```bash
pip install "machaon[mcp]"
python -m machaon.mcp_server
```

2 世代の SDK をサポート：`mcp` 1.x は `FastMCP`、2.x は `MCPServer` に改名。デコレータと `run(transport="stdio")` の表面は同じで、`create_server()` がインストール済みのものを選びます。

CI はサーバーをサブプロセスとして起動し、実際のクライアントでツールを呼びます（`scripts/mcp_smoke.py`）。トランスポートが壊れればビルドが失敗します。

## クライアント設定

Codex：

```bash
codex mcp add machaon -- python -m machaon.mcp_server
```

Claude Desktop（`claude_desktop_config.json`）：

```json
{
  "mcpServers": {
    "machaon": {
      "command": "python",
      "args": ["-m", "machaon.mcp_server"]
    }
  }
}
```

ファイルレベルサーバーは stdio で動き、モデル状態を持たないので再起動コストが低い。モデルレベルツールは明示的に開いた 1 つの `Session` を共有します。

## ファイルレベルツール

| ツール | 内容 |
|---|---|
| `describe_checkpoint(path)` | tensor 数、パラメータ数、dtype 構成、シャード数——ヘッダーのみ |
| `list_tensors(path, contains, suffix, layer, limit)` | 名前、形状、dtype。値は返さない |
| `audit_edit(base_path, edit_dir, max_rel_norm)` | OpenWSP v1 編集を base と照合し、ドリフトを再計算 |
| `apply_delta(base_path, tensor, delta_path, out_dir)` | 1 つの `.npy` delta を 1 tensor に適用し、新しいチェックポイントを書く |
| `gate(metrics)` | ゲートを適用。欠損指標は失敗 |

各ツールは `machaon.tools` の普通の関数に委譲し、テストはその関数を直接検証します。トランスポートは振る舞いを追加しません。

## モデルレベルツール

1 つのロード済み session で動きます。`open_model` が唯一の高コスト呼び出しです。

| ツール | 内容 |
|---|---|
| `open_model(path, dtype)` | モデルを session にロード |
| `model_status()` | パラメータ、層数、編集済み tensor、予算使用 |
| `capture_baseline(neutral_prompts, corpus_texts)` | **編集前**に参照分布を記録 |
| `propose_edit(prompt, target, layer, solver, steps, max_rel_norm)` | 編集を解き、書き込まずにコストを報告 |
| `apply_edit()` | 保留中の提案を書き込み、予算を原子的に課金 |
| `evaluate(prompts, targets)` | 精度、KL、baseline 比のパープレキシティ |
| `rollback()` | 編集済み tensor をすべて復元 |

想定ループは `machaon.session` と同じ：

```
open_model -> capture_baseline -> propose_edit -> apply_edit -> evaluate
                                                        |
                                                    rollback
```

`propose_edit` が書き込み前に相対 delta を返すのが要点です。agent は高すぎると判断して適用しない選択ができます。

## エラー形状

Session ツールは例外ではなく `{"error": "no model is open; call open_model first"}` を返すので、agent は汎用の `Error executing tool` ではなく理由を見られます。

## 設計境界

- **ファイルレベルはヘッダー優先。** 検査、監査、チェックポイント書き換えはモデルをメモリに載せません。
- **モデルレベルは明示的。** `open_model` が唯一の高コスト呼び出しで、`propose_edit` は `apply_edit` の前にコストを報告します。
- **編集位置は agent が決める。** 層選択は agent または `machaon.procedures.rome` の仕事で、サーバーはトランザクションと予算を強制します。
- **パスはプロセス権限を使う。** shell と同じ信頼境界で実行してください。詳細は [SECURITY.md](https://github.com/S0ra-ai/Machaon/blob/main/SECURITY.md)。
