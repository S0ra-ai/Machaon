# 重みプログラム

重みプログラムは、完全でレビュー可能な重み空間操作を記述する小さなテキストファイルです。

- どのチェックポイントから始めるか；
- モデルにどの事実を教えるか；
- どのプロンプトを変えてはいけないか；
- モデルをどれだけ変更してよいか；
- 保存後に再読み込みして検証するか。

`machaon first`、`machaon teach-many`、Web コンソールと同じループを共有できる形にしたものです。
プログラムは第二の編集エンジンを追加しません。パスとポリシーを解決し、教学チェーンを呼び、
実際に書き出されたバイトを検査します。

## プログラムを生成する

形式を手書きする必要はありません。`machaon program-new` は 1 つ以上の教学文から
レビュー可能なファイルを生成し、`machaon program` と同じパーサーで検証します。

```bash
machaon program-new ./hello.machaon \
  --teach "The Eiffel Tower is in the city of -> Rome" \
  --protect "Two plus two equals"
```

`--model` を省略すると `./models/SmolLM2-135M` から開始し、`fetch: smol135m` を追加して
`./models/SmolLM2-135M-machaon` へ出力します。`--model` と `--output` で独自パスを
指定できます。カスタムモデルには暗黙のダウンロードはありません。`--fetch` で明示的に
starter を追加し、`--no-fetch` で既定の starter を抑止できます。`--teach` と
`--protect` は繰り返し指定でき、パス `-` は標準出力へプログラムを出します。既存ファイルは
`--force` なしでは置換しません。

生成、確認、実行:

```bash
machaon program-new ./hello.machaon --teach "The capital of France is -> Paris"
machaon program ./hello.machaon --plan
machaon program ./hello.machaon
```

高度なフィールドは生成後のテキストで直接編集できます。生成器は一般的な経路だけを
カバーします。

## 完全な例

`hello.machaon` として保存します。

```text
version: 1
name: Hello, weight space

model: ./models/SmolLM2-135M
fetch: smol135m
output: ./models/hello-weight-space

budget: 0.10
max_rel_norm: 0.02
verify: true

teach: The Eiffel Tower is in the city of -> Rome
teach: The capital of France is -> Paris

protect: The capital of Germany is
protect: Water freezes at a temperature of
```

モデルをダウンロード・ロードせずに、解決済みの契約を確認できます。

```bash
machaon program hello.machaon --plan
```

実行します。

```bash
machaon program hello.machaon
# 監査記録をチェックポイント外に置く場合:
machaon program hello.machaon --report-out ./reports/hello-progress.json
```

`model` が存在せず `fetch` が starter を指定している場合だけダウンロードします。
各事実は独立に検索・局所検証され、最後に全体として安全なチェックポイントが
`<output>/final` に保存されます。`report.json` と
`<output>/program-report.json` も書き出されます。

`verify: true` は保存後にチェックポイントを再読み込みし、すべての事実を再検査します。
保存自体は成功して再検証に失敗した場合、状態は `verification_failed` になり終了コードは非ゼロです。
バイトを別途検証する場合にだけ `--no-verify` を使ってください。

ローカル Web コンソールの **重みプログラムを実行** パネルも同じ契約を使います。
プログラムを貼り付け、相対パスの基準フォルダーを設定して、**プログラムを確認** または
**プログラムを実行** を選びます。ブラウザー経由でも既存のファイル許可リストを通過します。
パネルから保存済みレポートを開き、`machaon program-verify` と同じ整合性検証または
保存バイト検証を実行することもできます。

## フィールド

必須は `model`、`output`、1 つ以上の `teach` だけです。

| フィールド | 既定値 | 意味 |
|---|---|---|
| `model` | 必須 | チェックポイントディレクトリ。相対パスはプログラムファイルから解決 |
| `output` | 必須 | 新しいチェックポイントのコンテナ。非空の既存ディレクトリは拒否 |
| `fetch` | なし | `model` がない場合に取得する starter: `smol135m`、`qwen05b`、`gpt2` |
| `teach` | 必須、複数可 | 教学文、または JSON の `prompt` / `answer` オブジェクト |
| `protect` | なし、複数可 | ベースライン・投影・パネルに使う無関係なプロンプト |
| `budget` | `0.10` | 全事実の累積相対ノルム予算 |
| `max_rel_norm` | `0.02` | 事実ごとの相対ノルム上限 |
| `max_kl` | `0.05` | 保護パネルの KL p95 上限 |
| `min_panel_agreement` | `0.875` | top-1 を保つ必要がある保護プロンプトの割合 |
| `layers` | 自動深さウィンドウ | `18,20,22` または `18-22` |
| `solver` | `direction` | `direction` または `gradient` |
| `dtype` | `float32` | `auto`、`float32`、`float16`、`bfloat16` |
| `quantize` | なし | `int4` または `int8`。`dtype: auto` が必要 |
| `device` | ランタイム既定 | `cpu`、`mps`、`cuda` など |
| `verify` | `true` | 保存したチェックポイントを再読み込みして全事実を検査 |
| `continue_on_reject` | `false` | 拒否された事実を記録して続行 |
| `keep_steps` | `false` | 安全な final だけでなく各ステップも保持 |

`#` はコメント、空行は無視されます。`key: value` または `key = value` を使い、
`teach` と `protect` は複数回書けます。

## 相対パス

`model` と `output` の相対パスは、プログラムファイルがあるディレクトリから解決されます。
プログラムと `models/` を別のマシンにコピーしても契約が変わらないようにするためです。

`--out DIR` はその実行の出力だけを上書きします。プログラムのハッシュと宣言された出力は
レポートに残るため、共有された契約を黙って書き換えることはできません。

```bash
machaon program hello.machaon --out ./models/experiment-2
```

## JSON 形式

同じ契約を JSON として生成できます。事実は文字列、`{"say": "..."}`、
または明示的な `{"prompt": "...", "answer": "..."}` です。

```json
{
  "version": 1,
  "model": "./models/SmolLM2-135M",
  "fetch": "smol135m",
  "output": "./models/hello-weight-space",
  "facts": [
    "The Eiffel Tower is in the city of -> Rome",
    {"prompt": "The capital of France is", "answer": " Paris"}
  ],
  "protect": ["Two plus two equals"],
  "verify": true
}
```

どちらの形式も同じ `program-report.json` を生成し、プログラムの SHA-256、
解決済みポリシー、各教学ステップ、検証結果を含みます。共有・レビュー時の監査記録です。

## レポートを信頼せずに検証する

保存されたレポートは、アドレス欄を除いた正規化文書から計算した `report_sha256` を
自身に含みます。コピー、コミット、issue への添付、ブラウザからのダウンロードの後でも
次のように検証できます。

```bash
machaon program-verify ./models/hello-weight-space/program-report.json
machaon program-verify ./models/hello-weight-space/program-report.json \
  --program hello.machaon
machaon program-verify ./models/hello-weight-space/program-report.json \
  --check-model
```

1 つ目はレポートのアドレスと必須構造を検証します。2 つ目はプログラムファイルのハッシュが
レポート内のプログラムアドレスと一致することも要求します。3 つ目はレポートが指す保存済み
チェックポイントを再読み込みし、全事実を再実行して、実測結果が記録された判定と一致するかを
示します。

チェックポイントを移動した場合は、新しい場所を明示できます。

```bash
machaon program-verify ./reports/lesson.json \
  --check-model --checkpoint ./copied/lesson/final
```

SHA-256 は内容アドレスであり署名ではありません。偶発的または無断の変更を検出しますが、
バイトを変えた後に誰でも再計算できます。`--check-model` は宣言した事実を保存済みの重みで
実行するため、より強い検証です。アドレス欄を持たない古いレポートは拒否され、検証済みとして
扱われません。

## 主張しないこと

局所チェックは、宣言した事実が成立し保護パネルが範囲内に収まったことを示します。
ベンチマークの代替ではありません。プログラムレポートはこの編集についての証拠であり、
モデル全体についての一般論ではありません。
