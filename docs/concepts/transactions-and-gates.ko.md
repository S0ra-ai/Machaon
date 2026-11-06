# 트랜잭션과 게이트

## 트랜잭션

되돌릴 수 없는 편집은 편집이 아니라 배포입니다.

```python
ws = Workspace()
ws.register("layer.8.mlp.c_proj", W)

txn = ws.begin("eiffel tower -> rome")
txn.apply("layer.8.mlp.c_proj", delta)
txn.commit(verdict)
```

- `apply`는 쓰기 전에 원본 배열을 복사합니다. 같은 tensor에 대한 두 편집은 역순으로 되돌아갑니다.
- `rollback()`은 모든 원본 바이트를 정확히 복원합니다.
- `commit(verdict)`는 `verdict.passed`가 false일 때 롤백합니다. `None`을 넘기면 무조건 커밋합니다. 아무것도 측정하지 않고 위험을 받아들일 때만 쓰세요.

닫힌 트랜잭션은 이후 작업을 거부합니다. 두 번 커밋하거나 롤백 후 apply할 수 없습니다.

## 게이트

게이트는 편집이 커밋될 자격을 얻는 방법입니다.

| gate | 임계값 | 보호하는 것 |
|---|---|---|
| `rewrite_acc` | ≥ 0.95 | 편집이 적용됨 |
| `paraphrase_acc` | ≥ 0.60 | 정확한 prompt 밖으로 일반화 |
| `unrelated_acc` | ≥ 1.00 | 다른 답이 변하지 않음 |
| `kl_p95` | ≤ 0.05 | 무관한 prompt의 분포 드리프트 |
| `kl_max` | ≤ 0.50 | 단일 치명적 이상치 없음 |
| `mmlu_delta` | ≥ −0.01 | 능력 보존 |
| `ppl_ratio` | ≤ 1.02 | 전역 손상 없음 |
| `longform_contradictions` | = 0 | 장문 생성 일관성 |

## 페일 클로즈

```python
from machaon.hygieia import evaluate

verdict = evaluate(metrics)
verdict.passed      # 지표가 빠지면 False
verdict.missing     # ['mmlu_delta', ...]
```

누락 지표는 실패입니다. 그렇지 않으면 스위트를 통과하는 가장 쉬운 길은 측정을 멈추는 것이고, 측정을 선택사항으로 만든 도구는 결국 측정 없이 쓰입니다.

좁은 실험에서는 명시적 게이트 부분집합을 넘길 수 있지만, 그렇다고 밝히세요:

```python
evaluate(metrics, {"rewrite_acc": (">=", 0.99), "ppl_ratio": ("<=", 1.05)})
```

## 누적 예산

한 번은 싸고, 스무 번은 싸지 않습니다. Machaon은 tensor별, 모델 전체로 `Σ‖ΔWᵢ‖ / ‖W‖`를 추적하고 초과 커밋을 거부합니다.

```python
from machaon import Workspace
from machaon.budget import EditBudget

budget = EditBudget(max_total=0.10)
ws = Workspace(budget=budget)
ws.register("layer.20.mlp.down_proj", W)

txn = ws.begin("edit 10")
txn.apply("layer.20.mlp.down_proj", delta)
txn.commit()          # BudgetExceeded 발생, 롤백, 아무것도 청구하지 않음
```

삼각부등식에 의해 이 합은 `‖ΣΔWᵢ‖ / ‖W‖`의 상한이므로 보수적입니다: 안전하게 강제할 수 있고 결코 미화하지 않습니다.

GPT-2 124M에서 각 편집 rel.Δ ≈ 0.011 실측:

| 편집 수 | 누적 Δ | WikiText-2 perplexity |
|---|---|---|
| 0 | 0.000 | 1.000 |
| 8 | 0.089 | 1.049 |
| 16 | 0.151 | 1.098 |
| 32 | 0.255 | 1.191 |

예산 `0.10`은 10번째 편집에서 거부합니다. 예산은 **commit** 시 청구되므로 `rollback()`과 실패한 verdict는 무료입니다.

## verdict 읽기

```
FAIL
  missing: longform_contradictions
  kl_p95 = 0.113 violates <= 0.05
```

두 가지 다른 문제입니다. `missing`은 측정하지 않음, 다른 줄은 측정했지만 실패. 이 순서로 고치세요.
