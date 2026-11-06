# 로드맵

"가장 쉽게 출시"가 아니라 "가중치 수술을 가장 유용하게" 순서입니다. 완료 항목에는 항상 테스트나 실측이 있습니다.

## 출시됨

- [x] 유도 가능한 불변량을 가진 선형대수 프리미티브
- [x] 비트 단위 롤백이 있는 트랜잭션 적용
- [x] 내용 주소 기반, 읽기 검증 delta 저장소
- [x] 페일 클로즈 게이트, KL 분위, bootstrap 구간
- [x] OpenWSP v1 편집 포맷, 바이트 일치, 적합성 테스트
- [x] safetensors 어댑터: 단일 파일, 디렉터리, 샤드 index
- [x] HF 런타임: 아키텍처 독립 레이어 발견, 방향 정규화, padding 정확 채점
- [x] 단일 함수 ROME procedure
- [x] provenance가 있는 벤치마크 로더(zsRE, WikiText-2, MMLU)
- [x] 편집되지 않은 모델로 채점하는 장문 일관성 스위트
- [x] 실측 비용 곡선이 있는 누적 편집 예산
- [x] Qwen2.5 0.5B, 1.5B, 3B, 7B 단계
- [x] Llama-3.1-8B의 int8과 int4
- [x] 메모리 효율 그래디언트 풀이(`densify_from`)
- [x] config에서 6개 아키텍처 패밀리를 인스턴스화해 테스트
- [x] MCP 서버: 5개 파일 수준 + 7개 모델 수준 도구, CI에서 실제 stdio
- [x] 영어·중국어·일본어·한국어 4개 언어 문서, GitHub Pages 자동 배포
- [x] 릴리스 흐름: sdist, wheel, GitHub release, 선택적 PyPI 배포

## 현재

- [ ] **MEMIT 스타일 배치 풀이.** 많은 사실, 하나의 제약 시스템, 하나의 예산. 단일 편집 경로가 블록.
- [ ] **L1 행동 공간: 샌드박스된 Python 편집 함수.** tensor 위 선언 함수를 같은 트랜잭션과 게이트 계약으로 실행.
- [ ] **더 많은 실측 단계.** 13B/14B 체크포인트 경로와 재현 가능한 메모리/처리량 표.
- [ ] **더 깊은 장문 스위트.** 멀티턴, 엔티티 일관성, 1k+ token 모순 탐지.

## 다음

- [ ] 평가 하네스 기반 레이어 선택 탐색
- [ ] OpenWSP writer의 희소·구조화 편집 형식
- [ ] 재현 가능한 편집 실행 공개 리더보드
- [ ] Apple Silicon과 CUDA 네이티브 벤치마크 매트릭스

## 인접 도구

Machaon의 단위는 기존 모델에 대한 표적 편집입니다. 근처 작업에는 성숙한 도구가 있고, 그 산출물은 이 계약과 조합됩니다:

- **병합과 task arithmetic:** mergekit delta는 `machaon.tools`와 `machaon.verify`로 감사·적용할 수 있습니다.
- **프루닝과 재배선:** 구조 편집은 delta로 도착해 같은 예산과 게이트를 통과합니다.
- **파인튜닝과 LoRA:** 동일 예산 LoRA baseline은 [`RESULTS.md`](RESULTS.md#r9-equal-budget-lora-baseline)에 실측되어 있고, adapter delta는 rank-one 편집과 비교할 수 있습니다.
- **추론 서빙:** `Checkpoint.write_with`로 편집된 체크포인트를 내보내고 어떤 Hugging Face 호환 스택에서도 서빙할 수 있습니다.
