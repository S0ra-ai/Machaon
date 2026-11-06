# 使用 OpenWSP

Machaon 逐字节实现 [OpenWSP edit format v1](https://github.com/openwsp/openwsp/blob/main/docs/edit-format.md)。

## 读取别人的编辑

```python
from machaon.interop import openwsp
from machaon.verify import audit_edit

manifest, tensors = openwsp.read_edit("path/to/edit", originals)
deltas = openwsp.delta_from_edit(manifest, tensors, originals)

report = audit_edit(deltas, originals, protected=protected_keys)
assert report["passed"]
```

`read_edit` 在返回任何内容前校验两个内容地址和每个 tensor 的哈希。截断或被改过的 payload 会被拒绝，而不是被应用。

## 写出别人能读的编辑

```python
manifest = openwsp.export_edit(originals, edited, "out/edit",
                               base_model="Qwen/Qwen3.5-0.8B",
                               revision="<pinned revision>")
manifest["edit_sha256"]
```

## 为什么重要

兼容性不是重点。重点是**一次编辑可以在不信任生产工具的情况下被审计**：

```python
report = check_protected(base, delta, protected)
```

它对任何 delta 都有效——我们的、别人的、手写的 patch。

## 一致性测试

测试套件对照参考实现生成的 fixture，而不是对照一份描述：

```
tests/fixtures/openwsp_v1/     manifest.json, edit.npz, originals/
tests/test_openwsp_interop.py  payload_sha256 和 edit_sha256 必须完全一致
```

如果上游地址变了，测试会失败，而不是悄悄漂移。

## 信任模型

- 地址承诺的是**内容**。作者身份和下游效果是另外两个断言，分别由签名和验收门建立。
- 导入会校验 manifest 中列出的每个 tensor。若想让检查点其余部分也进入同一次审计，把它们一并放进 `originals`。
