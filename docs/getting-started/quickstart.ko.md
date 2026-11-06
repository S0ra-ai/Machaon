# 빠른 시작

5분, 다운로드 없음. 여기 있는 모든 명령은 오프라인으로 실행됩니다.

## 1 · 데모 실행

```bash
python -m machaon.cli demo
```

```
op                         rel.delta   target err   preserve drift probe drift p95
----------------------------------------------------------------------------------
rank1 (naive)                 0.0509     1.60e-16         8.71e-02          0.0902
constrained (no cap)          0.0730     2.02e-16         3.82e-16          0.1394
constrained (capped)          0.0200     2.84e-01         2.52e-16          0.0382
```

세 번째 열을 보세요. 순진한 편집은 보호 key를 8.7e-2 움직이고, 제약된 편집은 3.8e-16에 머뭅니다.

## 2 · 내 행렬 편집하기

```python
import numpy as np
from machaon import rank1_constrained
from machaon.verify import check_protected

rng = np.random.default_rng(0)
W = rng.normal(0, 0.5, size=(64, 48))      # (d_out, d_in)
keys = rng.normal(size=(16, 48))           # 움직이면 안 됨
k = rng.normal(size=48)                    # 기억해야 함
v = W @ k + rng.normal(0, 1.0, size=64)    # 출력해야 함

W_new, delta = rank1_constrained(W, k, v, preserve=keys, max_rel_norm=0.05)

print(check_protected(W, delta, keys))
```

```
{'passed': True, 'max_drift': 3.1e-16, ...}
```

## 3 · 트랜잭션에 넣기

```python
from machaon import Workspace

ws = Workspace()
ws.register("layer.8.mlp.c_proj", W)

txn = ws.begin("first edit")
txn.apply("layer.8.mlp.c_proj", delta)

# 불리언 `.passed`를 가진 객체면 무엇이든 verdict가 될 수 있음
txn.commit()          # 또는 txn.rollback()
```

`commit(verdict)`는 `verdict.passed`가 false일 때 거부하고 원본 바이트를 복원합니다.

## 4 · 실제 모델에서 실행

```bash
export HF_ENDPOINT=https://hf-mirror.com
python examples/gpt2_rome.py
```

```
model      : gpt2 (12 layers)
edit       : transformer.h.8.mlp.c_proj.weight
delta      : rel 0.0153
  rewrite_acc  1
  paraphrase_acc 1
  kl_p95       0.000614724
  ppl_ratio    1.00079
PASS (5 gates)
```

## 다음

- [사실 편집하기](../guides/edit-a-fact.md) —— 전체 절차를 단계별로.
- [편집 평가하기](../guides/evaluate.md) —— 게이트, KL, 벤치마크.
- [수학 원리](../MATH.md) —— 유도와 실패 모드.
