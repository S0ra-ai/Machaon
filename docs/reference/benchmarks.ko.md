# 벤치마크

여기 모든 숫자는 라이선스와 SHA-256을 기록한 공개 데이터셋에서 나옵니다. 파일은 `scripts/fetch_benchmarks.py`가 받고 옆에 `.provenance.json`을 씁니다.

## 데이터셋

| 데이터셋 | 소스 | 라이선스 | 항목 | SHA-256 |
|---|---|---|---|---|
| zsRE(development) | `hf://datasets/wangzn2001/zsre/structured_zeroshot-dev-new_annotated_final.jsonl` | 미러 미선언; 업스트림 zsRE, Levy et al. 2017 | 10,813 | `05bbeb0ed8c4…` |
| WikiText-2 raw(test) | `hf://datasets/Salesforce/wikitext/wikitext-2-raw-v1/test` | CC-BY-SA-3.0 AND GFDL | 4,358 | `5f1bea067869…` |
| MMLU(test) | `hf://datasets/cais/mmlu/all/test` | MIT | 14,042 | `74a41822ce7d…` |

!!! warning "zsRE 라이선스"
    여기서 쓰는 미러는 라이선스를 선언하지 않습니다. zsRE는 지식 편집 문헌(MEND, SERAC, MEMIT)에서 널리 쓰이지만, 재배포한다면 원 소스에서 조건을 확인하세요.

언제든 검증:

```python
from machaon.bench import read_provenance, verify
verify("~/.cache/machaon/bench/zsre-dev.jsonl")
```

## 프로토콜

**Target injection.** 각 zsRE 질문마다 데이터셋 자체 엔티티 어휘에서 방해 token을 고르고, 모델이 그것을 예측하도록 편집한 뒤 원 질문과 사람이 쓴 바꿔쓰기를 측정합니다.

왜 "모델이 틀리는 항목 수리"가 아닌가? Qwen2.5-0.5B는 zsRE 첫 token 300개 중 299개를 맞히므로 그 프로토콜에는 개선 여지가 거의 없습니다. Target injection은 약한 모델과 강한 모델 모두에서 작동합니다.

**WikiText-2 — perplexity.** 편집 전후 남겨둔 문서의 평균 NLL을 재고 `exp(Δ mean NLL)`을 보고.

**MMLU — 능력.** 사지선다를 문자 완성의 평균 log-probability로 채점.

**Solver.** `gradient` 200 스텝, 모델 파라미터 동결, backward는 value 벡터 그래디언트만 누적. CPU에서 얻은 두 발견:

| 관찰 | 측정 |
|---|---|
| macOS CPU bf16 backward | 4.66초/스텝 |
| macOS CPU float32 backward | **0.18초/스텝**(26배 빠름) |
| Qwen2.5-0.5B 12층, 편집 1회 | rel. delta 0.150 |
| 같은 모델 20층 | rel. delta **0.0043**(35배 저렴) |

둘 다 solver 하이퍼파라미터보다 중요합니다. CPU에서는 float32를 쓰고, 행동을 바꿀 수 있으면서 출력에 가장 가까운 레이어를 고르세요.

## 결과

같은 프로토콜, 같은 8회 편집, 같은 평가기.

| 모델 | 파라미터 | 레이어 | 평균 rel.Δ | rewrite | rephrase(513) | WikiText-2 | MMLU |
|---|---|---|---|---|---|---|---|
| GPT-2 124M | 124M | 8 / 12 | 0.0135 | 0.500 | 0.259 | 1.066 | 0.220 → 0.220(n=100) |
| Qwen2.5-0.5B | 494M | 20 / 24 | 0.0133 | 0.375 | 0.275 | 1.030 | 0.440 → 0.440(n=50) |
| Qwen2.5-1.5B | 1544M | 20 / 28 | **0.0065** | 0.500 | 0.273 | **1.005** | 0.400 → 0.400(n=50) |
| Qwen2.5-3B | 3086M | 34 / 36 | 0.0072 | 0.625 | **0.327** | 1.009 | 0.500 → 0.500(n=30) |

1.5B는 100 스텝, 3B는 34층에서 30 스텝. 추세가 중요합니다: 0.5B에서 3B로, 같은 injection의 상대 delta는 약 절반, perplexity 비용은 3.0%에서 0.9%로 줄어듭니다. 3B 체크포인트는 샤드(2 파일)이며 `AutoModelForCausalLM`으로 로드해 레이어마다 `down_proj` 하나를 편집합니다.

### 1.5B 파일 수준 쓰기

```
source   : 338 tensors, 1543.7M params, 1 shard(s)
target   : model.layers.20.mlp.down_proj.weight [1536, 8960]
delta    : rel 0.0050
wrote    : /tmp/qwen15b_edited3 (2.3s)
untouched: model.embed_tokens.weight identical=True
```

2.9 GB 체크포인트를 2.3초에 다시 쓰고, tensor 하나만 바뀌고 나머지는 바이트 일치. 7–8B용 경로로, 모델과 두 번째 복사본을 동시에 메모리에 둘 수 없을 때 씁니다.

## 32 GB 노트북의 7B

Qwen2.5-7B는 7.6B 파라미터, 4샤드 체크포인트. **int8 가중치**로 32 GB 머신에 올라가고 상주 약 9.4 GB, forward 2.5초.

**파일 수준 쓰기**는 한 샤드를 다시 쓰고 나머지를 하드링크:

```
source   : 339 tensors, 7615.6M params, 4 shard(s)
target   : model.layers.24.mlp.down_proj.weight [3584, 18944]
delta    : rel 0.0030
shard    : model-00004-of-00004.safetensors (linked 3, copied 0)
wrote    : /tmp/qwen7b_edit (2.3s)
untouched: lm_head.weight identical=True
```

피크 메모리는 샤드 하나(약 4 GB)이며 체크포인트 전체가 아닙니다.

**전체 평가**는 같은 int8 모델에서 실행: `optimum-quanto`가 약 9.4 GB로 유지하고, `HFModel.densify_from(layer)`가 편집 레이어와 그 이후를 float32로 바꿔 backward가 28개 블록 중 2개만 통과합니다.

| solver | 평균 rel.Δ | rewrite | rephrase | WikiText-2 | 피크 RSS |
|---|---|---|---|---|---|
| `direction`(backward 없음) | 0.0503 | 0.750 | 0.667 | 1.260 | 10.6 GB |
| **`gradient`, 국소** | **0.0043** | **1.000** | **0.929** | **1.007** | 10.2 GB |

설정: 28층 중 26층, 30 스텝, lr 0.05, int8 가중치에서 `lm_head` 양자화 제외, 편집 3회, 각 15 바꿔쓰기.

### 출력까지의 깊이

7B 설정으로 3B(36층 중 26층, 10블록 남음)를 다시 실행:

| 3B 설정 | 이후 블록 | 평균 rel.Δ | rewrite | rephrase | WikiText-2 |
|---|---|---|---|---|---|
| 26층, lr 0.02, 80 스텝 | 10 | 0.0052 | 0.625 | 0.230 | 1.002 |
| 26층, lr 0.05, 30 스텝 | 10 | 0.0079 | 0.500 | 0.236 | 1.002 |
| 32층, lr 0.05, 30 스텝 | 4 | 0.0068 | 0.625 | 0.312 | 1.009 |
| **34층, lr 0.05, 30 스텝** | **2** | 0.0072 | 0.625 | **0.327** | 1.009 |

깊이는 일반화를 돕지만(0.230 → 0.327) 8개 타깃에서 평평해집니다.

**동일 타깃 수 실행이 해석을 수정합니다.** 7B 실험과 같은 3개 타깃, 42개 바꿔쓰기로 3B는 거의 같은 숫자에 도달합니다:

| 모델 | 편집 수 | 평균 rel.Δ | rewrite | rephrase | WikiText-2 |
|---|---|---|---|---|---|
| Qwen2.5-3B, 34층 | 3 | 0.0082 | **1.000** | **0.857**(36/42) | 1.0068 |
| Qwen2.5-7B, 26층(int8) | 3 | 0.0043 | **1.000** | **0.929**(39/42) | 1.0070 |

그래서 이전 0.625는 8개 타깃 샘플의 성질(마지막 5개가 더 어려움)이지 단단한 용량 한계가 아닙니다. 같은 샘플 크기에서 주된 차이는 비용입니다: 3B는 같은 결과에 약 2배의 상대 delta가 필요합니다.

## 8B: 현재 최대 실측 모델

Llama-3.1-8B는 8,030.3M 파라미터, 4샤드로 선언한 상한 안. 32 GB 머신에서 두 구성:

| 구성 | 평균 rel.Δ | rewrite | rephrase | WikiText-2 | 피크 RSS |
|---|---|---|---|---|---|
| int8, 31/32층(1블록 밀집) | 0.0072 | **1.000** | 0.500 | 1.284 | 10.2 GB |
| int4, 30/32층(2블록 밀집) | 0.0138 | 0.500 | 0.500 | **1.031** | 8.6 GB |

각 2회 편집, 20 바꿔쓰기, WikiText 2문서, MMLU n=8.

**int8**은 밀집 블록 하나로 10.2 GB에서 rewrite 1.000. **int4**는 밀집 블록 둘로 8.6 GB에서 perplexity 1.031.

**7.6B int8 + 2블록 윈도우**가 rewrite 1.000 / rephrase 0.929 / perplexity 1.007로 현재 가장 강한 조합입니다.

`--quantize int4`는 quanto unpack kernel용 `ninja`가 PATH에 필요합니다.

## 왜 그냥 파인튜닝하지 않나

`examples/lora_baseline.py`는 같은 레이어, 같은 3개 prompt, 같은 타깃 token, 같은 30 그래디언트 스텝으로 rank-8 adapter를 학습합니다. Qwen2.5-0.5B, float32:

| 방법 | rewrite | rephrase | WikiText-2 | MMLU(n=8) |
|---|---|---|---|---|
| rank-1 폐형식, 30 스텝 | 1.000 | 0.976(41/42) | **1.044** | 0.500 → 0.625 |
| LoRA rank 8, 30 스텝 | 1.000 | **1.000** | **1.441** | 0.500 → **0.250** |

LoRA는 표현력이 더 커 남겨둔 바꿔쓰기를 1.000으로 만듭니다. 대신 46,080개의 무제약 파라미터를 쓰고 모델을 44% perplexity와 25점 MMLU 하락으로 끌고 갑니다. rank-one 편집은 보호 key에 투영되고 예산으로 제한되어 10분의 1 손상으로 거의 같은 유효성에 도달합니다.

## 누적 비용

`examples/budget_sweep.py`는 편집을 하나씩 적용하고 체크포인트에서 남겨둔 perplexity를 측정합니다. GPT-2 124M, 8층, 100 스텝:

| 편집 수 | 누적 Δ | WikiText-2 perplexity |
|---|---|---|
| 0 | 0.000 | 1.000 |
| 8 | 0.089 | 1.049 |
| 16 | 0.151 | 1.098 |
| 32 | 0.255 | 1.191 |

`EditBudget(max_total=0.10)`은 10번째 편집에서 거부. 누적 상대 변화가 약 0.1을 넘으면 124M–0.5B 모델이 눈에 띄게 열화된다는 것이 기본 예산의 경험적 근거입니다.

## 장문 일관성

`examples/longform_check.py`는 Eiffel Tower 편집을 적용하고 5개 prompt의 120 token greedy 생성을 비교합니다. 일관성은 **편집되지 않은** 모델로 채점.

| 모델 | baseline 반복 | 편집 후 반복 | Δ | NLL ratio | 빈 생성 |
|---|---|---|---|---|---|
| GPT-2 124M(120 tok) | 0.621 | 0.578 | **−0.043** | 1.089 | 0.0 |
| Qwen2.5-1.5B(80 tok) | 0.097 | 0.196 | **+0.099** | 0.938 | 0.0 |

GPT-2에서는 반복이 거의 그대로, 1.5B에서는 약 두 배가 되어 0.10 게이트에 정확히 걸립니다. 이것은 경고이지 축하할 통과가 아닙니다: 임계값이 느슨할 수 있고 80 token은 짧습니다.

## 재현

```bash
export HF_ENDPOINT=https://hf-mirror.com
python scripts/fetch_benchmarks.py

HF_HUB_OFFLINE=1 python examples/bench_edits.py \
  --model gpt2 --layer 8 --edits 8 --dtype float32 --solver gradient

HF_HUB_OFFLINE=1 python examples/bench_edits.py \
  --model /path/to/Qwen2.5-0.5B --layer 20 --edits 8 \
  --dtype float32 --solver gradient --mmlu 50 --wikitext 20
```

## 다음 측정

1. 구성마다 100+ 편집과 분산 추정.
2. 모델 횡단 누적 예산 스윕.
3. 각 규모의 장문 일관성.
4. 3B와 7B LoRA baseline.
