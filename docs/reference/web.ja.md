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
machaon web --allowed-root /models --port 8765
# machaon-web も同じランチャーです
```

起動時に一度だけ使う token を含む URL が表示されます。サーバーは既定で
`127.0.0.1` にだけバインドされます。`--allowed-root` または
`MACHAON_WEB_ALLOWED_ROOTS` で checkpoint のルートを指定してください。
`torch`、`transformers`、`safetensors` のいずれかがない場合、最初の session が
黙って失敗する代わりに、正確な `pip install "machaon[hf]"` コマンドを banner で
表示します。

## 表示言語

コンソールは English、簡体字中国語、日本語、한국어 に対応しています。上部の
言語ピッカーは静的コピーだけでなく、ジョブ状態、指標、エラーも切り替え、
選択を `localStorage` に保存します。`?lang=zh`、`?lang=ja`、`?lang=ko` の
直リンクも使えます。指定がない場合はブラウザー言語に従い、未対応なら English
にフォールバックします。

## 初回ガイド

初回認証時に、**FIRST EDIT** のゼロからのカードを先に示し、編集の目的、
検査 → セッションを開く → baseline → 提案 → 適用 →
評価 → 保存/ロールバックの流れ、各コントロールの意味を説明する短いガイドが
開きます。**探索を始める** で既読になり、上部の **ガイド** からいつでも
再表示できます。Hero 下の 4 ステップ表示は軽いリマインダーとして残ります。

## 最初の編集カード

初回認証時には **FIRST EDIT** カードも表示されます。チェックポイント未選択なら
カード内でスターターを選べます（既定は SmolLM2-135M、多言語向けは
Qwen2.5-0.5B、ほかに GPT-2）。取得して開き、
"The Eiffel Tower is in the city of -> Rome" をプリフィルし、教学レシピを実行して
結果を表示します。2 つの例ボタンは、編集が適用される `Eiffel → Rome` と、
モデルがすでに知っている `France → Paris` を読み込むので、入力せずに両方の結果を
確認できます。session が開くとカードは消え、受け入れられた編集は **保存** 用に
session に残ります。教学が適用されると、カードには
**層、相対 Δ、panel 保持数、KL p95** が表示され、最初の成功が高度なワークスペースと
同じ証拠語彙を持ちます。さらに `<model>-machaon` が入った
**チェックポイント出力** 欄と **チェックポイントを保存** ボタンが残り、右列と同じ
`/api/session/save` を使います。保存後の **保存済みチェックポイントを開く** は
編集済み session を閉じて（メモリ上のバイトをロールバック）保存済みを開き、次の
事実用のきれいな session を残します。**完了** は現在の session を保持します。開いた後はまず、
教えた prompt/answer を開き直したチェックポイントで自動的に評価し、保存後の
回答が再現するかを表示します。その後、文をクリアし、出力を新しい
チェックポイントの次の番号付き `<name>-machaon-N`（例: `<name>-machaon-2`）に更新して
**別の事実を教える** を提示するので、
メインフォームに戻らず同じループを続けられます。保存ごとに新しい session と
ベースラインが始まるため、事実をまたぐ累積予算と元 panel のドリフト検査には
**複数の事実を教える** を使ってください。**隠す** は `localStorage` に保存されます。

最後に正常に開いた、または保存したチェックポイントのパスも `localStorage` に
保存されます。再読み込み時に Checkpoint 欄へ戻すため、意図せず別の starter を
ダウンロードしません。FIRST EDIT の状態にも使用するチェックポイントが表示されます。

保存後の **複数の事実を教える** は多事実パネルを開き、現在の事実をプリフィルして
`<model>-machaon-many` コンテナを提案します。リスト結果は状態チップだけでなく、
各事実の prompt、状態、層、Δ、全体ドリフトを展開します。

## 保存した事実を確認

Checkpoint 列の **保存した事実を確認** パネルは、session を開かず、変更せずに
選択中チェックポイントの prompt/answer、または 1 行 1 事実のリストを評価します。
`/api/check` を呼び、保存済み重みを読み直し、`machaon check` と同じ前スペース
処理で各事実の PASS/FAIL、回答、精度を表示します。

## 重みプログラムを実行

Checkpoint 列の **重みプログラムを実行** パネルは、`machaon program` と同じ
`.machaon` テキストまたは JSON を受け付けます。**プログラムフォルダ** を許可
ルート内のディレクトリにすると、プログラム内の相対 `model` / `output` パスは
そこから解決されます。**プログラムを確認** は `/api/program/plan` を呼び、
モデルを読み込まずにモデル、事実、予算、検証方針を表示します。
**プログラムを実行** はバックグラウンドジョブで `/api/program/run` を呼び、
`<output>/program-report.json` を書いた後、保存済みチェックポイントを再読み込み
してから報告します。ジョブバナーは各事実の step/total と状態、最後に結果と保存
バイト検証を表示します。

**現在のチェックポイントを使用** は開いている session または選択中の
チェックポイント、教学文、preserve prompts からエディターを埋めます。
**.machaon をダウンロード** はエディター内容を可搬ファイルとして保存するため、
ブラウザーで成功したレシピをそのまま共有できます。**.machaon を開く** はローカルの
テキスト/JSON ファイルをエディターへ読み込み、確認・実行時には同じ許可ルート制約を
適用します。
2 つの組み込みテンプレートは既存チェックポイント不要です。**SmolLM2 最初の編集** は
1 つの事実を、**2 つの事実** は同じ予算・検証方針で 2 事実の連鎖を実行します。
**プログラムを確認** または実行開始した直近 8 件はブラウザーのローカル履歴に残り、
**読み込む** で復元できます。アップロードは行わず、`localStorage` だけに保存します。
**プログラムをコピー** はエディター内容を、**実行レポートをダウンロード** は現在の
実行 JSON を保存し、**出力チェックポイントを使う** は final を新しい session で開きます。
**事実からプログラムを構築** は 1 行 1 教学文を、現在のチェックポイント、出力命名規則、
標準予算と検証方針を使ったレビュー可能な `.machaon` に変換します。

どちらのルートも他のコンソール操作と同じファイル許可リストを適用します。
貼り付けたプログラムだけで許可ルート外の読み書きはできません。

## 三列のワークフロー

1. **Checkpoint**: safetensors の header を読み、tensor 数、パラメータ数、
   dtype、shard 数を表示します。ブラウザーに重みの値は送りません。
   チェックポイントがない場合は starter を選んで **Download starter** を押すと、
   許可ルートへミラー経由も試してダウンロードし、パスを埋めます。ジョブ
   バナーには SSE で端点、ファイル、バイト/パーセント進捗が表示されます。
   **ダウンロード後に開く** を有効にすると session まで自動で開きます。
2. **編集契約**: session を開き、baseline を取得し、層を指定して proposal を
   解きます。preserve prompts は零空間制約になり、量子化モデルの gradient
   solve では **Densify from edit layer** を有効にします。proposal は書き込み
   前に相対変化と target probability を報告します。
3. **証拠**: proposal を apply し、baseline に対して evaluate します。
   fail-closed gate の判定と時系列イベントを確認でき、rollback は元の
   tensor バイトを復元します。
   永続化するときは Applied session の delta を新しい checkpoint ディレクトリへ
   書き出し、元のディレクトリは変更しません。

### ひとつの文で教える

編集列の上部にある **ONE SENTENCE** 欄が最短経路です。`machaon intent` と同じ
文法を受け付け、`machaon teach` と同じ教学レシピを実行します。

1. **Preview parse** は文をコンパイルし、prompt/answer 欄へ反映します。
2. **Teach automatically** は深さウィンドウを検索し、編集を preserve prompts の
   零空間へ投影し、最良の候補を適用します。目標、予算、panel、KL の全チェックが
   通ったときだけ保持し、拒否された候補は自動でロールバックします。

任意の **Also write the edited checkpoint** 欄に出力先を書くと、許可ルート内の
新しいディレクトリへ保存します。空欄なら session 内に保持し、後から **Save** を
使えます。

**複数の事実を教える** も同じ欄にあります。1 行 1 文で受け付け、事実ごとに
検索を実行し、`<out>/final` と `<out>/report.json` を出力コンテナに書き出します。
現在の session は base のまま残り、**最終チェックポイントを使う** で最後の
全体安全な結果を開けます。

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
