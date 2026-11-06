# 영공간 투영

이 프로젝트의 토대가 되는 하나의 아이디어입니다.

## 문제

key `k`를 value `v`로 매핑하는 rank-one 편집은 단일 제약의 최소 노름 해입니다:

```
ΔW = (v − W k) kᵀ / (kᵀ k)
```

`k`에는 정확하지만 다른 입력에 대해서는 아무것도 보장하지 않습니다. 실제로는 이웃 key를 움직여 손상을 줍니다.

## 수정

`K = [k₁ … kₙ]ᵀ`를 바꾸고 싶지 않은 key, `Q`를 `span(K)`의 정규직교 기저라고 합시다. 투영 `P = I − Q Qᵀ`를 정의하고, 투영된 key를 따라 편집을 씁니다:

```
k⊥  = P k
ΔW  = (v − W k) k⊥ᵀ / (k⊥ᵀ k)
```

그러면:

- **타깃**: `ΔW k = v − W k`, 즉 `W' k = v`. 정확.
- **보호**: 모든 보호 key에 대해 `ΔW kᵢ = 0`. 기계 정밀도로 정확.

분자는 **원래** `k`를 씁니다. 양쪽에 `k⊥`를 쓰는 것은 흔한 버그로, 값은 들어가지만 불필요한 노름을 씁니다.

## 구현 메모

핫 패스에서 `d_in × d_in` 투영 행렬을 실체화하지 않습니다:

```python
Q = nullspace_basis(preserve)          # (d_in, r)
k_perp = project_complement(k, Q)      # k - Q Qᵀ k, O(d_in · r)
```

`d_in = 4096`, `r = 32`이면 128 MB 밀집 행렬 대신 0.5 MB 기저로 충분합니다.

## 실패하는 경우

**타깃 key가 보호 부분공간에 들어간다.** 이때 `k⊥ ≈ 0`이고 두 제약을 동시에 만족하는 편집은 없습니다. Machaon은 조용히 열화하지 않고 오류를 냅니다:

```
ValueError: target key lies in the preserved subspace: the edit cannot be made
without changing protected behaviour
```

이것은 버그가 아니라 신호입니다: 보호 집합을 줄이거나 다른 key를 고르세요.

## 보장의 범위

보장은 **1차이고 보호 집합에서는 정확**합니다: 제공한 벡터에 대해 `ΔW kᵢ = 0`이 기계 정밀도로 성립합니다. 주변 입력은 1차로 보존되고, 나머지는 사후 검사로 측정합니다: key에는 [`verify`](../reference/api.md), 무관한 prompt에는 KL.

칼날 폭이 알려진 메스로 다루세요. "다른 것은 아무것도 움직이지 않았다"는 증명은 아닙니다.
