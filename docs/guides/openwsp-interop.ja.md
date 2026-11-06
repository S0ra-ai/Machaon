# OpenWSP を使う

Machaon は [OpenWSP edit format v1](https://github.com/openwsp/openwsp/blob/main/docs/edit-format.md) をバイト単位で実装しています。

## 他人の編集を読む

```python
from machaon.interop import openwsp
from machaon.verify import audit_edit

manifest, tensors = openwsp.read_edit("path/to/edit", originals)
deltas = openwsp.delta_from_edit(manifest, tensors, originals)

report = audit_edit(deltas, originals, protected=protected_keys)
assert report["passed"]
```

`read_edit` は何かを返す前に 2 つの内容アドレスと tensor ごとのハッシュを検証します。切り詰められた、または改変された payload は適用されず拒否されます。

## 他人が読める編集を書く

```python
manifest = openwsp.export_edit(originals, edited, "out/edit",
                               base_model="Qwen/Qwen3.5-0.8B",
                               revision="<pinned revision>")
manifest["edit_sha256"]
```

## なぜ重要か

互換性が目的ではありません。目的は**編集が生成ツールを信頼せずに監査できる**ことです：

```python
report = check_protected(base, delta, protected)
```

これはどんな delta にも効きます：ours、theirs、手書きの patch。

## 適合性

テストスイートは説明ではなく、参照実装が生成した fixture と比較します：

```
tests/fixtures/openwsp_v1/     manifest.json, edit.npz, originals/
tests/test_openwsp_interop.py  payload_sha256 と edit_sha256 が完全一致すること
```

上流のアドレスが変われば、静かにドリフトせずテストが失敗します。

## 信頼モデル

- アドレスが約束するのは**内容**です。作者性と下流効果は別の主張で、それぞれ署名とゲートで確立します。
- インポートは manifest に列挙された全 tensor を検証します。チェックポイントの残りも同じ監査に入れたい場合は `originals` に含めてください。
