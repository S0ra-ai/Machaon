# Web コンソール

ローカル Web コンソールは、Python API、CLI、MCP サーバーと同じ
`machaon.tools` / `machaon.session` をブラウザから使うための薄い入口です。
モデル処理を複製せず、三つの入口が同じトランザクション契約を共有します。

```text
ブラウザ ──> FastAPI / SSE ──> machaon.tools / machaon.session
                                  │
                    CLI ──────────┤
                    MCP server ───┘
```

## インストールと起動

コアは NumPy だけで動作します。Web コンソールには追加依存を入れます。

```bash
pip install "machaon[web]"
machaon-web --allowed-root /models --port 8765
```

起動時に一度だけ使う token を含む URL が表示されます。サーバーは既定で
`127.0.0.1` にだけバインドされます。`--allowed-root` または
`MACHAON_WEB_ALLOWED_ROOTS` で checkpoint のルートを指定してください。

## 三列のワークフロー

1. **Checkpoint**: safetensors の header を読み、tensor 数、パラメータ数、
   dtype、shard 数を表示します。ブラウザーに重みの値は送りません。
2. **編集契約**: session を開き、baseline を取得し、層を指定して proposal を
   解きます。preserve prompts は零空間制約になり、量子化モデルの gradient
   solve では **Densify from edit layer** を有効にします。proposal は書き込み
   前に相対変化と target probability を報告します。
3. **証拠**: proposal を apply し、baseline に対して evaluate します。
   fail-closed gate の判定と時系列イベントを確認でき、rollback は元の
   tensor バイトを復元します。
   永続化するときは Applied session の delta を新しい checkpoint ディレクトリへ
   書き出し、元のディレクトリは変更しません。

モデル読み込み、勾配 solve、評価は単一 worker で実行され、SSE で進捗を
配信します。プロセスごとに一つのモデル session を持ちます。

適用後の評価が終わったら **Persist** で現在の delta を新しい checkpoint に
書き出します。単一 tensor は `Checkpoint.write_tensor`、複数は
`Checkpoint.write_with` を使います。元の checkpoint は変更せず、出力先は
許可ルート内である必要があります。

## 大規模チェックポイント例：Qwen3.6-27B

コンソールはモデル非依存で、同じコントロールがすべての対応ファミリーに
使えます。27B は大規模チェックポイントの一例です。34 GB の統合メモリ環境では
`int4` または `int8` を使い、言語デコーダーの後半を編集点にします。

```text
path:      ./qwen36-27b
dtype:     auto
quantize:  int4
device:    cpu
layer:     60
budget β:  0.01
module:    mlp.down_proj
solver:    gradient
densify:   on
```

最初に header だけで checkpoint を確認してください。Transformers runtime、
単一 shard 書き込み、`Checkpoint.write_tensor` の詳細は
[Qwen3.6-27B ガイド](../guides/qwen27b.md) にあります。

## 運用ルール

コンソールは loopback に置き、許可ルートを実際に編集する checkpoint の
ディレクトリだけに絞ります。表示された token は session と checkpoint 書き込み
を承認する一時的な能力です。Web 層は任意の Python コードを受け取らず、パス
許可リスト、予算検査、トランザクション rollback を通します。
