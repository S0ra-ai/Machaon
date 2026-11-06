# 관련 연구

이 아이디어는 다섯 커뮤니티의 교차점에 있습니다. 어느 것도 전체 루프를 제공하지 않으며, Machaon은 편집·트랜잭션·수용을 하나의 조작 가능한 전체로 연결합니다.

## 파라미터 수준 지식 편집

| 연구 | 메모 |
|---|---|
| [ROME](https://arxiv.org/abs/2202.05262) | rank-one 사실 편집. 이 노선의 시작 |
| [MEMIT](https://arxiv.org/abs/2210.07229) | 수천 사실 배치 편집 |
| [MEND](https://arxiv.org/abs/2110.11309) | 편집을 예측하는 하이퍼네트워크 |
| [SERAC](https://arxiv.org/abs/2206.06520) | 가중치 대신 외부 메모리 |
| [Survey](https://arxiv.org/abs/2310.19704) | 문헌 최고의 입구 |
| [EasyEdit](https://github.com/zjunlp/EasyEdit) | 가장 가까운 기존 프레임워크(2.9k★). 배치 스크립트 지향 |

## 학습 없는 가중치 수술

| 연구 | 메모 |
|---|---|
| [Model Surgery](https://arxiv.org/abs/2407.08770) | 아주 적은 파라미터로 행동이 바뀜 |
| [SCALPEL](https://arxiv.org/abs/2601.07411) | 능력 절제를 위한 저랭크 파라미터 편집 |
| [ThinkEdit](https://arxiv.org/abs/2503.22048) | 가중치 편집이 추론 길이를 바꿈 |
| [Resisting Harmful Instructions through Training-Free Weight Editing](https://arxiv.org/abs/2610.21189) | 실용적이며 동시에 위험하다는 증거 |
| [TARS](https://arxiv.org/abs/2412.10257) | 지식 제거를 위한 표적 각도 반전 |

## 표현·활성화 편집

| 연구 | 메모 |
|---|---|
| [Refusal is mediated by a single direction](https://arxiv.org/abs/2406.11717) | "하나의 방향, 하나의 행동" 결과 |
| [RepE](https://arxiv.org/abs/2310.01405) | 표현 공학 |
| [CAA](https://arxiv.org/abs/2312.06681) | 대비 활성화 덧셈 |
| [ActAdd](https://arxiv.org/abs/2306.03341) | 활성화 공학 |
| [SAELens](https://github.com/decoderesearch/SAELens) | 희소 오토인코더 특징을 편집 대상으로 |

## 가중치 공간 대수

| 연구 | 메모 |
|---|---|
| [Task Arithmetic](https://arxiv.org/abs/2212.04089) | task vector 덧셈으로 모델 편집 |
| TIES / DARE | 간섭 인식 병합 |
| [mergekit](https://github.com/arcee-ai/mergekit) | 성숙한 툴체인(7.4k★) |
| [Evolutionary model merge](https://github.com/SakanaAI/evolutionary-model-merge) | 병합 레시피 탐색 |

## LLM 기반 최적화

| 연구 | 메모 |
|---|---|
| [OPRO](https://arxiv.org/abs/2309.03409) | LLM을 optimizer로 사용 |
| [TextGrad](https://arxiv.org/abs/2406.07496) | 텍스트 그래디언트 |
| [SEAL](https://arxiv.org/abs/2506.10943) | 모델이 자체 업데이트 지시를 생성 |
| [Foundation models understand weights](https://arxiv.org/abs/2503.00838) | "LLM이 가중치 공간을 추론할 수 있다" 지지 |
| [LLM as a Tool, Not an Agent](https://arxiv.org/abs/2604.16555) | agentic editing에 대한 가장 날카로운 반론 |
| [Agentic NAS](https://arxiv.org/abs/2607.07984) | 반대 진영 |

## 만들기 전에 읽을 두 편

1. [Revisiting Parameter-Based Knowledge Editing: Theoretical Limits and Empirical Evidence](https://arxiv.org/abs/2606.00570) —— 파라미터 편집에 근본적 상한이 있다고 논증.
2. [Long-form evaluation of model editing](https://arxiv.org/abs/2402.09394) —— 짧은 QA를 통과한 편집이 장문 생성에서 무너짐.

둘 다 `machaon.hygieia` 게이트 설계의 내력벽입니다.

## 빈틈

1. agent가 조작할 수 있는 편집 행동 공간이 없고 기존은 Python API나 스크립트.
2. 스냅샷·검증·롤백을 맡는 트랜잭션 계층이 없음.
3. 평가가 일등 시민이 아님.
4. LLM이 **수술을 계획**하는 시스템이 없음: 진단 → 부위 선택 → 제약 선언 → 폐형식 solver로 전달.
