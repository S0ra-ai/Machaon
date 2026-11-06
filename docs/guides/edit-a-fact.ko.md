# 사실 편집하기

`machaon.procedures.rome` 구현과 같은 전체 절차입니다.

## 여섯 단계

### 1 · 행렬 선택

사실은 보통 중간 레이어의 MLP 블록에 있습니다. 깊이의 60–70%에서 시작하고, 능력이 손상되면 더 얕은 레이어로 옮깁니다.

```python
from machaon.runtime import HFModel

runtime = HFModel(model, tokenizer)
name = runtime.edit_target(layer=8)          # transformer.h.8.mlp.c_proj.weight
```

`edit_target`은 모델 패밀리 사이에서 이름을 번역하고(`down_proj` ⇄ `c_proj`), 가중치 방향을 정규화해 수학 레이어가 항상 `(d_out, d_in)`을 보게 합니다.

### 2 · key 캡처

key는 행렬이 **실제로** 보는 입력이며, 마지막 layer norm 전의 residual stream이 아닙니다.

```python
ids = runtime.encode(["The Eiffel Tower is located in the city of"])[0]
key = runtime.capture_module_input("transformer.h.8.mlp.c_proj", ids)[0]
```

사실이 하나의 템플릿에 묶여 있지 않다면 여러 바꿔쓰기에서 평균하세요.

### 3 · 보호 집합 구성

```python
neutral = ["The capital of Germany is", "Water freezes at a temperature of", ...]
preserve = np.stack([runtime.capture_module_input(module, runtime.encode([p])[0])[0]
                     for p in neutral])
```

움직이면 안 되는 key입니다. 보통 8~30개면 충분하고, 보호가 많을수록 유효성 비용이 커집니다.

### 4 · value 풀이

value는 행렬이 그 key에 대해 내야 할 출력입니다. ROME은 짧은 그래디언트로 최적화합니다. 대상은 **단일 벡터**이며 모델 전체가 아닙니다:

```python
result = rome_edit(runtime, prompt, target, layer=8,
                   preserve_prompts=neutral, steps=250)
```

### 5 · 예산과 함께 적용

```python
W_new, delta = rank1_constrained(W, key, value, preserve=preserve,
                                 max_rel_norm=0.02)
```

예산은 `‖ΔW‖/‖W‖`의 상한입니다. 더 필요하면 초과하지 않고 덜 전달합니다.

### 6 · 검증

```python
from machaon.verify import check_protected
check_protected(W, delta, preserve)     # max_drift는 ~1e-16
```

## 엔드투엔드 실행

```bash
export HF_ENDPOINT=https://hf-mirror.com
python examples/gpt2_rome.py
```

```
model      : gpt2 (12 layers)
edit       : transformer.h.8.mlp.c_proj.weight
solved p(t): 0.9968 after 250 steps
delta      : rel 0.0153
  rewrite_acc  1
  paraphrase_acc 1
  kl_p95       0.000614724
  ppl_ratio    1.00079
PASS (5 gates)
```

## 진단

| 증상 | 원인 | 해결 |
|---|---|---|
| 정확한 prompt는 되지만 바꿔쓰기 실패 | key가 한 템플릿에 묶임 | 바꿔쓰기 평균 |
| 능력 하락 | 레이어가 너무 중앙 | 2~4 레이어 얕게 |
| bf16에서 아무 변화 없음 | delta가 반올림으로 사라짐 | float64로 계산 |
| 무관한 prompt 드리프트 | 보호 집합 없음 | 추가 |
| "preserved subspace" 거부 | target key가 보호 span 안 | 집합 축소 |
