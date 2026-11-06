# 아키텍처

Machaon은 서로를 거의 모르는 세 계층입니다.

```
                    ┌──────────────────────────┐
   edit spec  ───►  │  machaon.linalg          │  순수 NumPy, 모델 없음
   (DSL, JSON)      │  rank1 · null-space · ...│
                    └───────────┬──────────────┘
                                │ delta (ndarray)
                    ┌───────────▼──────────────┐
                    │  machaon.transaction     │  비트 단위 롤백
                    │  machaon.store           │  내용 주소
                    └───────────┬──────────────┘
                                │ applied / rejected
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
┌───────▼───────┐     ┌─────────▼─────────┐   ┌─────────▼─────────┐
│ machaon.runtime│     │ machaon.hygieia   │   │ machaon.interop   │
│ HF / safetensors│    │ gates · runner    │   │ OpenWSP v1        │
└───────────────┘     └───────────────────┘   └───────────────────┘
```

## 두 계약

**Runtime.** `next_logits(items) -> (n, vocab)`와 `nll(items) -> (n,)`. 이것이 모델 대상 표면의 전부입니다. `HFModel`이 Hugging Face causal LM 위에 구현하고, 테스트 백엔드는 네 줄로 구현할 수 있습니다.

**Edit.** `(W, k, v, preserve) -> (W_new, delta)`. 순수 배열. 프레임워크 타입, device, dtype 의견이 없습니다: 내부는 모두 float64.

계약이 이렇게 좁아서 아키텍처 추가는 `LAYER_PATHS` 항목 하나, 편집 추가는 순수 함수 하나로 끝납니다.

## 수학 계층이 NumPy인 이유

- torch 없는 CI를 포함해 어느 머신에서나 테스트 가능;
- delta는 프레임워크 사이를 옮길 수 있음;
- 보장(`ΔW k_i = 0`)은 프레임워크가 아니라 숫자의 성질.

Torch는 `runtime/`과 `adapters/` 경계에만 나타납니다.

## 왜 인플레이스가 아니라 트랜잭션인가

롤백은 편집의 일부입니다. `Transaction`은 적용 전 원본 바이트를 복사하므로 `rollback()`이 정확히 복원합니다. `commit(verdict)`는 verdict가 통과하지 못하면 거부해 기본을 페일 클로즈로 만듭니다.

## 게이트가 페일 클로즈인 이유

누락 지표는 실패입니다. 그렇지 않으면 스위트를 통과하는 가장 쉬운 길은 측정을 멈추는 것이고, 측정을 선택사항으로 만든 도구는 결국 측정 없이 쓰입니다.

## 포맷 경계

`machaon.interop.openwsp`는 외부 포맷의 자기완결 구현입니다. 설명이 아니라 참조 구현이 생성한 산출물에 대해 적합성 테스트합니다. 업스트림 주소가 바뀌면 조용히 드리프트하지 않고 테스트가 실패합니다.

## 설계 경계

- **폐형식 우선.** 모델 가중치는 그래디언트 하강으로 최적화하지 않습니다. ROME은 하나의 value 벡터를 최적화하고 delta를 수학 계층에 넘깁니다.
- **텍스트는 경계.** 텍스트는 runtime tokenizer를 통해 들어오고, 편집 계층은 배열만 봅니다.
- **오케스트레이션은 호출자.** `Session`과 MCP 서버가 상태 기계(open → baseline → propose → apply → evaluate → rollback)를 제공하고 스케줄링은 agent나 스크립트에 남깁니다.
