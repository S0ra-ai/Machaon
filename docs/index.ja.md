# Machaon

**LLM のための直接的な重み手術 —— 勾配ではなく重みを編集する。**

*矢だけを抜き、身体はそのまま残す。*

Machaon はモデルのパラメータを直接変更し、その変更が他の能力を壊していないことを証明します。何を触れてよいか、実際に何が動いたか、間違っていたらどう戻すか。1 回の編集は diff・再生・ロールバックできる、内容アドレス指定の小さな成果物です。

## 3 つの保証

| 保証 | 実装 | 検証 |
|---|---|---|
| 保護した key は動かない | 零空間投影、`ΔW k⊥ = 0` | `verify.check_protected` |
| 拒否された編集は完全に戻る | 適用前に元のバイトをコピー | `transaction.rollback` |
| 測っていない編集は通らない | 欠損指標は失敗として扱う | `hygieia.evaluate` |
| 編集は再現できる | delta を内容アドレスで保存 | `store.DeltaStore` |

## インストール

```bash
git clone https://github.com/S0ra-ai/Machaon.git
cd Machaon

uv pip install -e .              # NumPy のみ：数学、ゲート、相互運用
uv pip install -e ".[hf]"        # + torch + transformers、実モデル用
uv pip install -e ".[bench]"     # + pyarrow、ベンチマーク用
uv pip install -e ".[mcp]"       # + MCP SDK、Python 3.10+
uv pip install -e ".[dev,docs]"  # テスト、lint、カバレッジ、ドキュメント
```

Python 3.9–3.13 をテストしています。`make ci` でオフライン検査を一括実行できます。

## 30 秒で試す

```python
import numpy as np
from machaon import rank1_constrained

W_new, delta = rank1_constrained(W, key, value, preserve=protected, max_rel_norm=0.02)
```

```python
from machaon.verify import check_protected

report = check_protected(W, delta, protected)
assert report["passed"] and report["max_drift"] < 1e-12
```

大事なのは 2 つ目のブロックです。1 つ目は数式、2 つ目は判定です。

## 実測規模

| モデル | 層 | 平均 rel.Δ | rewrite | rephrase | WikiText-2 |
|---|---|---|---|---|---|
| GPT-2 124M | 8 / 12 | 0.0135 | 0.500 | 0.259 | 1.066 |
| Qwen2.5-0.5B | 20 / 24 | 0.0133 | 0.375 | 0.275 | 1.030 |
| Qwen2.5-1.5B | 20 / 28 | **0.0065** | 0.500 | 0.273 | **1.005** |
| Qwen2.5-3B | 34 / 36 | 0.0072 | 0.625 | **0.327** | 1.009 |
| Qwen2.5-7B（int8） | 26 / 28 | **0.0043** | **1.000** | **0.929** | **1.007** |
| Llama-3.1-8B（int4） | 30 / 32 | 0.0138 | 0.500 | 0.500 | 1.031 |

## ドキュメント

- [インストール](getting-started/installation.md)
- [クイックスタート](getting-started/quickstart.md)
- [事実を編集する](guides/edit-a-fact.md)
- [編集を評価する](guides/evaluate.md)
- [ベンチマークと再現](reference/benchmarks.md)
- [Python API](reference/api.md)
- [MCP サーバー](reference/mcp.md)

## 状態

Alpha ですが、コアは実測済みです。数学、トランザクション、ストレージ、ゲート、ランタイム、safetensors アダプタ、OpenWSP v1 を実装し、Python 3.9–3.13 でテストしています。実モデルは GPT-2 124M、Qwen2.5 0.5B / 1.5B / 3B / 7B、Llama-3.1-8B をカバー。7B/8B は int8/int4 重みと局所的な密行列ウィンドウを使います。6 つの CI ジョブが lint、テスト行列、アダプタとカバレッジ、MCP トランスポート、ドキュメントビルド、wheel/sdist インストールを担当します。次は[ロードマップ](roadmap.md)へ。
