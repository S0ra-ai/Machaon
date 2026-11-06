# OpenWSP 사용하기

Machaon은 [OpenWSP edit format v1](https://github.com/openwsp/openwsp/blob/main/docs/edit-format.md)을 바이트 단위로 구현합니다.

## 다른 사람의 편집 읽기

```python
from machaon.interop import openwsp
from machaon.verify import audit_edit

manifest, tensors = openwsp.read_edit("path/to/edit", originals)
deltas = openwsp.delta_from_edit(manifest, tensors, originals)

report = audit_edit(deltas, originals, protected=protected_keys)
assert report["passed"]
```

`read_edit`은 무엇이든 반환하기 전에 두 콘텐츠 주소와 tensor별 해시를 검증합니다. 잘리거나 변조된 payload는 적용되지 않고 거부됩니다.

## 다른 사람이 읽을 수 있는 편집 쓰기

```python
manifest = openwsp.export_edit(originals, edited, "out/edit",
                               base_model="Qwen/Qwen3.5-0.8B",
                               revision="<pinned revision>")
manifest["edit_sha256"]
```

## 왜 중요한가

호환성이 목적이 아닙니다. 목적은 **편집을 생성 도구를 신뢰하지 않고 감사할 수 있다**는 것입니다:

```python
report = check_protected(base, delta, protected)
```

어떤 delta에도 작동합니다: 우리 것, 남의 것, 손으로 쓴 patch.

## 적합성

테스트 스위트는 설명이 아니라 참조 구현이 생성한 fixture와 비교합니다:

```
tests/fixtures/openwsp_v1/     manifest.json, edit.npz, originals/
tests/test_openwsp_interop.py  payload_sha256과 edit_sha256이 정확히 일치해야 함
```

업스트림 주소가 바뀌면 조용히 드리프트하지 않고 테스트가 실패합니다.

## 신뢰 모델

- 주소가 약속하는 것은 **내용**입니다. 저자성과 하류 효과는 별개의 주장이며 각각 서명과 게이트로 확립합니다.
- 가져오기는 manifest에 나열된 모든 tensor를 검증합니다. 체크포인트의 나머지도 같은 감사에 넣고 싶다면 `originals`에 포함하세요.
