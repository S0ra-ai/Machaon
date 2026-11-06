# 編集を評価する

クリーンな checkout から再現できない数値を報告しないでください。

## Runner

`EditSuite` が何を測るかを記述し、runner が測ります。どちらもバックエンド非依存です。

```python
from machaon.hygieia import EditSuite, evaluate_edit, judge

suite = EditSuite(
    target_items=["The Eiffel Tower is located in the city of"],
    target_token_ids=[tokenizer(" Rome").input_ids[0]],
    paraphrase_items=[...],                   # base がすでに " Paris" と答える prompt
    paraphrase_token_ids=[...],
    neutral_items=[...],                      # KL パネル
    corpus_items=[...],                       # パープレキシティパネル
)

metrics = evaluate_edit(before, after, suite)
verdict = judge(before, after, suite)         # 既定ゲートを適用
```

## 指標の定義

| 指標 | 定義 |
|---|---|
| `rewrite_acc` | 次の token が期待値である割合 |
| `paraphrase_acc` | 同上、留保した言い換えで |
| `kl_p95` | 中立 prompt 上の `KL(P_before ‖ P_after)` の 95 パーセンタイル |
| `kl_max` | 最悪の単一中立 prompt |
| `ppl_ratio` | 留保テキスト上の `exp(mean NLL_after − mean NLL_before)` |

KL は平均ではなく分位数で報告します：壊れた 1 つの prompt は、無傷の 100 個より重要です。

## ベンチマークパネル

アドホックな prompt はツールが動くかを教え、ベンチマークはモデルがまだ動くかを教えます。

```bash
python scripts/fetch_benchmarks.py        # ファイルごとに provenance を記録
python examples/gpt2_bench.py --edits 8
```

| パネル | データセット | ライセンス |
|---|---|---|
| 編集有効性 | zsRE development | [ベンチマーク](../reference/benchmarks.md)参照 |
| パープレキシティ | WikiText-2 raw test | CC-BY-SA-3.0 AND GFDL |
| 能力 | MMLU test | MIT |

## 出力の読み方

```
== zsRE (edits: 8, layer 8, mean rel.delta 0.0114) ==
  rewrite   0.000 -> 0.500
  rephrase  0.000 -> 0.889   (513 held-out paraphrases)

== WikiText-2 (held out, 20 documents) ==
  perplexity ratio  1.0806

== MMLU (100 questions) ==
  accuracy  0.220 -> 0.220   (delta +0.000)
```

- 有効性は本物：8 個中 4 個の rewrite、513 個中 456 個の言い換えが反転。
- コストも本物：パープレキシティが 8% 上昇。
- GPT-2 の MMLU は粗い信号。1.5B+ のパネルはより細かい差を解像します。報告はするが、読みすぎないこと。

## 長文一貫性

短答指標は最もよくある失敗を見落とします：正しく答えているのに生成が崩れる。

```python
from machaon.hygieia import LONGFORM_GATES, longform_report
from machaon.hygieia.gates import evaluate

report = longform_report(pristine, edited, prompts, max_new_tokens=120)
evaluate(report, LONGFORM_GATES)
```

| 指標 | 定義 |
|---|---|
| `longform_nll_ratio` | 編集モデルのテキストを**元モデル**で評価した NLL / baseline NLL |
| `longform_repetition_delta` | 編集後の 4-gram 反復率 − baseline |
| `longform_empty_rate` | 空生成の割合 |

judge model も追加ダウンロードも不要。局所編集なら ratio は 1.0 付近、反復 delta は 0.0 付近に留まるはずです。

## 私たちが守るルール

1. base model が**間違える**項目だけを編集し、改善を選択バイアスにしない。
2. base model がすでに正しく答える言い換え集合を使う。
3. すべての数値の隣に provenance（データセット、revision、ライセンス、SHA-256）を記録する。
4. 改善しなかった部分も報告する。
