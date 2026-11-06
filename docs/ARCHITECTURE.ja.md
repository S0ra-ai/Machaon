# アーキテクチャ

Machaon は互いをほとんど知らない 3 層です。

```
                    ┌──────────────────────────┐
   edit spec  ───►  │  machaon.linalg          │  純 NumPy、モデルなし
   (DSL, JSON)      │  rank1 · null-space · ...│
                    └───────────┬──────────────┘
                                │ delta (ndarray)
                    ┌───────────▼──────────────┐
                    │  machaon.transaction     │  ビット単位ロールバック
                    │  machaon.store           │  内容アドレス
                    └───────────┬──────────────┘
                                │ applied / rejected
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
┌───────▼───────┐     ┌─────────▼─────────┐   ┌─────────▼─────────┐
│ machaon.runtime│     │ machaon.hygieia   │   │ machaon.interop   │
│ HF / safetensors│    │ gates · runner    │   │ OpenWSP v1        │
└───────────────┘     └───────────────────┘   └───────────────────┘
```

## 2 つの契約

**Runtime。** `next_logits(items) -> (n, vocab)` と `nll(items) -> (n,)`。これがモデル向け表面のすべてです。`HFModel` が Hugging Face causal LM 上に実装し、テストバックエンドは 4 行で実装できます。

**Edit。** `(W, k, v, preserve) -> (W_new, delta)`。純粋な配列。フレームワーク型、device、dtype の意見はありません：内部はすべて float64。

契約がこれだけ狭いので、アーキテクチャ追加は `LAYER_PATHS` の 1 エントリ、編集追加は 1 つの純関数で済みます。

## 数学層が NumPy である理由

- torch のない CI を含む任意のマシンでテストできる；
- delta はフレームワーク間で移植できる；
- 保証（`ΔW k_i = 0`）はフレームワークではなく数の性質。

Torch は `runtime/` と `adapters/` の境界にだけ現れます。

## なぜインプレースではなくトランザクションか

元に戻せない編集は編集ではなくデプロイです。`Transaction` は適用前に元バイトをコピーするので、`rollback()` が正確に復元します。`commit(verdict)` は verdict 未通過時に拒否し、既定をフェイルクローズにします。

## ゲートがフェイルクローズである理由

欠損指標は失敗です。そうでなければスイートを通す最短経路は測定をやめることになり、測定を任意にしたツールは最終的に測定なしで使われます。

## フォーマット境界

`machaon.interop.openwsp` は外部フォーマットの自己完結実装です。説明ではなく、参照実装が生成した成果物に対して適合性テストします。上流アドレスが変われば、静かにドリフトせずテストが失敗します。

## 設計境界

- **閉形式優先。** モデル重みは勾配降下で最適化しません。ROME は 1 つの value ベクトルを最適化し、delta を数学層に渡します。
- **テキストは境界。** テキストは runtime の tokenizer を通って入り、編集層は配列だけを見ます。
- **編成は呼び出し側。** `Session` と MCP サーバーが状態機械（open → baseline → propose → apply → evaluate → rollback）を提供し、スケジューリングは agent かスクリプトに残します。
