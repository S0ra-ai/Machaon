# 자주 묻는 질문

**이것은 파인튜닝인가요?**
아니요. 모델 가중치 위의 optimizer도 학습 데이터도 없습니다. 편집은 한 번 적용되는 폐형식 또는 저랭크 delta입니다. 유일한 그래디언트는 ROME 풀이 시 단일 value 벡터 위에 있습니다.

**GPT-2 외 모델에서 동작하나요?**
런타임은 Llama/Qwen/Mistral/Gemma(`model.layers`), GPT-2/GPT-J(`transformer.h`), GPT-NeoX(`gpt_neox.layers`), OPT, MPT의 레이어를 찾습니다. 두 가중치 방향을 모두 처리합니다. 실제 체크포인트는 GPT-2 124M, Qwen2.5 0.5B / 1.5B / 3B / 7B, Llama-3.1-8B를 포함하며 각 단계에서 같은 벤치마크 패널을 씁니다.

**왜 LoRA를 쓰지 않나요?**
같은 레이어, 같은 스텝 수 실측: rank-8 adapter는 rewrite 1.000 / rephrase 1.000에 도달하지만 44% perplexity와 25점 MMLU를 치릅니다. rank-one 편집은 1.000 / 0.976을 4% perplexity로 달성합니다. LoRA는 표현력이 더 크지만 제약이 없습니다. [R9](RESULTS.md#r9-equal-budget-lora-baseline) 참고.

**70B 모델을 편집할 수 있나요?**
Machaon의 실측 범위는 8B까지입니다. int8/int4 가중치와 국소 밀집 윈도우로 풀이를 10.2 GB 안에 유지합니다. 그 이상은 체크포인트 수준에서 tensor 하나씩 스트리밍할 수 있고, 평가 경로는 주어진 메모리에 따라 확장됩니다. 풀이 비용은 편집점 이후 윈도우에 달려 있고 모델 전체가 아닙니다.

**rank-one 편집으로 충분한가요?**
하나의 사실에는 대개 충분합니다. 행동에는 저랭크 + 방향(`bake_direction` 또는 `ablate_output`)이 보통 더 좋습니다. 대량 편집의 다음 단계는 MEMIT 스타일 배치 풀이이며 [로드맵](roadmap.md)에 있습니다. 오늘은 하나의 `EditBudget` 아래 사실별 편집을 조합하거나 저랭크 방향을 쓰세요.

**편집은 성공했는데 바꿔쓰기가 실패하는 이유는?**
key가 하나의 token 열에 묶여 있습니다. 여러 바꿔쓰기에서 key를 캡처해 평균하세요. 바꿔쓰기 집합도 base model이 원래 정답을 내는 것이어야 합니다. 그렇지 않으면 노이즈를 측정합니다.

**`rank1_constrained`가 "preserved subspace"를 내는 이유는?**
target key가 보호 key의 span 안에 들어가 두 제약이 양립하지 않습니다. 보호 집합을 줄이거나 다른 key를 고르세요.

**왜 평균이 아니라 `kl_p95`를 보고하나요?**
망가진 prompt 하나가 무사한 100개보다 중요합니다. 평균은 그것을 숨깁니다.

**왜 틀린 답을 고치는 대신 target injection인가요?**
Qwen2.5-0.5B가 zsRE 첫 token 300개 중 299개를 맞히기 때문입니다. 모델이 틀리는 항목만 편집하는 프로토콜은 강한 모델에서 개선 여지가 거의 없습니다. injection은 약한 모델과 강한 모델 모두에서 작동합니다.

**왜 bf16이 아니라 float32인가요?**
CPU에서 PyTorch bf16 backward는 병적으로 느립니다: 같은 0.5B 모델에서 4.66초/스텝 대 float32 0.18초/스텝. delta는 둘 다 float64로 계산하고, 모델 dtype은 value 풀이의 forward/backward에만 영향을 줍니다.

**레이어 선택이 왜 그렇게 중요한가요?**
12층 편집은 뒤의 12개 블록을 통과해야 합니다. Qwen2.5-0.5B에서 같은 편집이 12층 rel. Δ 0.150, 20층 0.0043 — 35배 차이입니다.

**Hugging Face에 접근할 수 없습니다.**
`python scripts/fetch_model.py --repo ... --dest ...`를 쓰면 ModelScope로 폴백합니다. 또는 `HF_ENDPOINT=https://hf-mirror.com`을 설정하세요. 스크립트는 전송 후 safetensors 헤더도 검증합니다.

**왜 반복률을 baseline 상대로 보고하나요?**
GPT-2는 편집 전부터 greedy decoding에서 루프를 돌기 때문입니다. 절대 임계값은 편집되지 않은 모델도 실패시켜 편집에 대해 아무것도 알려주지 않습니다.

**편집을 몇 번 쌓을 수 있나요?**
측정하세요. GPT-2에서 32회 rank-one 편집(누적 Δ 0.255)은 19% perplexity, 8회는 5%. `EditBudget(max_total=0.10)`은 대략 10번째에서 거부하며 권장 기본값입니다.

**지표가 빠지면 gate가 실패하는 이유는?**
그렇지 않으면 통과하는 가장 쉬운 길은 측정을 멈추는 것입니다. 페일 클로즈는 설계 선택이지 실수가 아닙니다.

**OpenWSP 없이 포맷만 쓸 수 있나요?**
예. `machaon.interop.openwsp`는 독립 구현이며 참조 구현이 만든 fixture에 대한 적합성 테스트를 거쳤습니다.

**이것은 안전 정렬을 제거하나요?**
방향 절제는 한 줄로 가능합니다. 그래서 프로젝트가 거부 규칙, 추가 전용 편집 기록, 게이트 스위트를 함께 제공합니다. [SECURITY.md](https://github.com/S0ra-ai/Machaon/blob/main/SECURITY.md) 참고.
