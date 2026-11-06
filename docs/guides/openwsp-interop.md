# Work with OpenWSP

Machaon implements the [OpenWSP edit format v1](https://github.com/openwsp/openwsp/blob/main/docs/edit-format.md)
byte for byte.

## Read someone else's edit

```python
from machaon.interop import openwsp
from machaon.verify import audit_edit

manifest, tensors = openwsp.read_edit("path/to/edit", originals)
deltas = openwsp.delta_from_edit(manifest, tensors, originals)

report = audit_edit(deltas, originals, protected=protected_keys)
assert report["passed"]
```

`read_edit` verifies both content addresses and every per-tensor hash before
returning anything. A truncated or edited payload is refused, not applied.

## Write an edit they can read

```python
manifest = openwsp.export_edit(originals, edited, "out/edit",
                               base_model="Qwen/Qwen3.5-0.8B",
                               revision="<pinned revision>")
manifest["edit_sha256"]
```

## Why it matters

Compatibility is not the point. The point is that **an edit can be audited
without trusting the tool that produced it**:

```python
report = check_protected(base, delta, protected)
```

works on any delta — ours, theirs, or a hand-written patch.

## Conformance

The test suite compares against a fixture produced by the reference
implementation, not against a description of it:

```
tests/fixtures/openwsp_v1/     manifest.json, edit.npz, originals/
tests/test_openwsp_interop.py  payload_sha256 and edit_sha256 must match exactly
```

If the upstream address changes, the test fails rather than silently drifting.

## Trust model

- The address commits to **content**. Authorship and downstream effect are
  separate claims, established by signatures and gates respectively.
- Import validates every tensor listed in the manifest. Add the rest of the
  checkpoint to `originals` when you want them inside the same audit.
