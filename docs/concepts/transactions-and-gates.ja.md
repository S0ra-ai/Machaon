# トランザクションとゲート

## トランザクション

元に戻せない編集は編集ではなく、デプロイです。

```python
ws = Workspace()
ws.register("layer.8.mlp.c_proj", W)

txn = ws.begin("eiffel tower -> rome")
txn.apply("layer.8.mlp.c_proj", delta)
txn.commit(verdict)
```

- `apply` は書き込む前に元の配列をコピーします。同じ tensor への 2 回の編集は逆順に戻ります。
- `rollback()` はすべての元バイトを正確に復元します。
- `commit(verdict)` は `verdict.passed` が false のときロールバックします。`None` を渡すと無条件にコミットします。何も測らずリスクを受け入れるときだけ使ってください。

閉じたトランザクションは以降の操作を拒否します。2 回コミットしたり、ロールバック後に apply したりはできません。

## ゲート

ゲートは編集がコミットされる資格を得る方法です。

| gate | しきい値 | 守るもの |
|---|---|---|
| `rewrite_acc` | ≥ 0.95 | 編集が効いた |
| `paraphrase_acc` | ≥ 0.60 | 正確な prompt の外へ汎化した |
| `unrelated_acc` | ≥ 1.00 | 他が変わっていない |
| `kl_p95` | ≤ 0.05 | 無関係な prompt の分布ドリフト |
| `kl_max` | ≤ 0.50 | 単一の壊滅的な外れ値がない |
| `mmlu_delta` | ≥ −0.01 | 能力が保たれた |
| `ppl_ratio` | ≤ 1.02 | 全体の損傷がない |
| `longform_contradictions` | = 0 | 長文生成が一貫している |

## フェイルクローズ

```python
from machaon.hygieia import evaluate

verdict = evaluate(metrics)
verdict.passed      # 指標が欠けていれば False
verdict.missing     # ['mmlu_delta', ...]
```

欠損指標は失敗です。そうでなければスイートを通る最短経路は測定をやめることになり、測定を任意にしたツールは最終的に測定なしで使われます。

狭い実験では明示的なゲート部分集合を渡せますが、そう明記してください：

```python
evaluate(metrics, {"rewrite_acc": (">=", 0.99), "ppl_ratio": ("<=", 1.05)})
```

## 累積予算

1 回は安く、20 回は安くありません。Machaon は tensor ごと、モデル全体で `Σ‖ΔWᵢ‖ / ‖W‖` を追跡し、超えるコミットを拒否します。

```python
from machaon import Workspace
from machaon.budget import EditBudget

budget = EditBudget(max_total=0.10)
ws = Workspace(budget=budget)
ws.register("layer.20.mlp.down_proj", W)

txn = ws.begin("edit 10")
txn.apply("layer.20.mlp.down_proj", delta)
txn.commit()          # BudgetExceeded を投げ、ロールバックし、何も消費しない
```

三角不等式により、この和は `‖ΣΔWᵢ‖ / ‖W‖` の上界なので保守的です：安全に強制でき、決して美化しません。

GPT-2 124M で各編集 rel.Δ ≈ 0.011 の実測：

| 編集数 | 累積 Δ | WikiText-2 パープレキシティ |
|---|---|---|
| 0 | 0.000 | 1.000 |
| 8 | 0.089 | 1.049 |
| 16 | 0.151 | 1.098 |
| 32 | 0.255 | 1.191 |

予算 `0.10` は 10 回目で拒否します。予算は **commit** 時に課金されるので、`rollback()` と失敗した verdict は無料です。

## verdict の読み方

```
FAIL
  missing: longform_contradictions
  kl_p95 = 0.113 violates <= 0.05
```

2 つの別問題です。`missing` は測っていない、もう一方は測って失敗した。この順に直します。
