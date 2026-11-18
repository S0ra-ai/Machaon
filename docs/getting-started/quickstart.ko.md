# 빠른 시작

5분. 처음 두 단계는 오프라인으로 실행됩니다. 교수 단계는 로컬
체크포인트를 사용합니다.

## 0 · 0에서 명령 하나로

```bash
machaon first
```

새 머신에서는 SmolLM2-135M(약 270 MB)을 받아 열고,
"The Eiffel Tower is in the city of -> Rome"을 가르친 뒤 수용된 편집을
`./models/SmolLM2-135M-machaon`에 저장합니다. `--say`로 사실을 바꾸고,
`--model`로 기존 체크포인트를 지정하며, `--no-save`는 메모리에만 유지합니다.
`--program-out first.machaon`을 추가하면 실행 전에 같은 내용을 이식 가능하고 검증 가능한
가중치 프로그램으로 저장할 수 있습니다.

## 1 · 특정 체크포인트 가르치기

체크포인트에서 검증된 편집까지 가장 짧은 경로는 명령 하나입니다.

0.5B 체크포인트를 한 번 받습니다(약 1 GB, 미러 자동 시도):

```bash
machaon fetch --starter qwen05b --dest ./models/Qwen2.5-0.5B
machaon fetch --list          # smol135m / qwen05b / gpt2
```

다운로더는 표준 라이브러리만 사용하고 각 safetensors 헤더를 검증하며 기존
파일을 재사용합니다.

```bash
machaon start
machaon intent "에펠탑은 -> 로마"

machaon teach --model ./models/Qwen2.5-0.5B \
  --say "에펠탑은 -> 로마"
```

`intent`는 모델을 읽지 않고 문장을 정확한 `prompt`와 `answer`로 컴파일합니다.
`teach`는 이어서:

1. 모델이 이미 답할 수 있는지 확인하고;
2. 무관한 기준 panel을 자동으로 캡처해 그 영공간으로 편집을 투영하고;
3. 깊이 방향의 후보 layer를 차례로 적용·평가·롤백하고;
4. 최선의 layer를 다시 풀어 예산 안에서 적용하고;
5. 목표 token, panel의 top-1 유지율, KL을 측정하고;
6. 필수 검사가 모두 통과할 때만 편집을 유지합니다. 실패하면 원본 바이트를
   복원합니다.

기본 출력은 `<checkpoint>-machaon`이라는 형제 디렉터리이며 원본
체크포인트는 변경되지 않습니다. `--dry-run`은 쓰지 않고 검색만,
`--no-save`는 메모리에만 유지하며, layer를 이미 안다면
`--layers 8,10,12`를 사용합니다.

지원하는 문장 형식:

```text
에펠탑은 -> 로마
prompt: 에펠탑은
answer:  로마
when asked "the Eiffel Tower is", answer " in Paris"
教会模型：北京的首都是 → 北京
```

화살표와 라벨 형식은 정확합니다. 일반 문장도 받지만 분할 지점이 가정이므로
중간 신뢰도로 표시됩니다. 로컬 검사는 목표와 선언한 panel만 다룹니다.
벤치마크 수준의 주장에는 전체 승인 보고서가 필요합니다.

교수는 답의 **첫 token**을 고정합니다. 여러 token 답은 경고로 표시되며, 나머지
연속은 벤치마크 질문이지 이 로컬 검사의 보장이 아닙니다.
### 여러 사실 가르치기

```bash
cat > facts.txt <<'EOF'
에펠탑은 -> 로마
프랑스의 수도는 -> 파리
EOF

machaon teach-many --model ./models/Qwen2.5-0.5B --facts facts.txt --out ./models/qwen-many
```

각 사실은 자체 layer 검색과 로컬 검사를 수행합니다. 체인은 마지막 전역 안전
체크포인트를 `<out>/final`에 두고 각 단계를 `<out>/report.json`에 기록합니다.
한 사실이 거부되면 이전 사실을 유지한 채 중지합니다.
`--continue-on-reject`는 실패를 건너뛰고, `--keep-steps`는 체인 전체를 보존합니다.


행렬, key, 보호 집합을 직접 고르고 싶다면 아래의 수동 단계를 사용하세요.

## 2 · 데모 실행

```bash
python -m machaon.cli demo
```

```
op                         rel.delta   target err   preserve drift probe drift p95
----------------------------------------------------------------------------------
rank1 (naive)                 0.0509     1.60e-16         8.71e-02          0.0902
constrained (no cap)          0.0730     2.02e-16         3.11e-16          0.1394
constrained (capped)          0.0200     2.84e-01         8.55e-17          0.0382
```

세 번째 열을 보세요. 순진한 편집은 보호 key를 8.7e-2 움직이고, 제약된 편집은 3.8e-16에 머뭅니다.

프로그램 전체 경로(생성, 실행, 저장, 재로드, 보고서 검증)를 오프라인으로 실행하려면:

```bash
python examples/tiny_program.py
```

로컬에 작은 랜덤 GPT-2를 만들고 아무것도 다운로드하지 않습니다. 이는 연결 경로 검사이며
언어 모델 성능 결과가 아닙니다.

## 3 · 내 행렬 편집하기

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

## 4 · 트랜잭션에 넣기

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

## 5 · 실제 모델에서 실행

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
- [수학 원리](../MATH.md) —— 유도, 진단, 적용 범위.
