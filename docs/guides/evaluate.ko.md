# 편집 평가하기

깨끗한 checkout에서 재현할 수 없는 숫자는 보고하지 마세요.

## Runner

`EditSuite`가 무엇을 측정할지 기술하고, runner가 측정합니다. 둘 다 백엔드 독립적입니다.

```python
from machaon.hygieia import EditSuite, evaluate_edit, judge

suite = EditSuite(
    target_items=["The Eiffel Tower is located in the city of"],
    target_token_ids=[tokenizer(" Rome").input_ids[0]],
    paraphrase_items=[...],                   # base가 이미 " Paris"라고 답하는 prompt
    paraphrase_token_ids=[...],
    neutral_items=[...],                      # KL 패널
    corpus_items=[...],                       # perplexity 패널
)

metrics = evaluate_edit(before, after, suite)
verdict = judge(before, after, suite)         # 기본 게이트 적용
```

## 지표 정의

| 지표 | 정의 |
|---|---|
| `rewrite_acc` | 다음 token이 기대값인 비율 |
| `paraphrase_acc` | 위와 같음, 남겨둔 바꿔쓰기에서 |
| `kl_p95` | 중립 prompt에서 `KL(P_before ‖ P_after)`의 95 백분위 |
| `kl_max` | 가장 나쁜 단일 중립 prompt |
| `ppl_ratio` | 남겨둔 텍스트에서 `exp(mean NLL_after − mean NLL_before)` |

KL은 평균이 아니라 분위로 보고합니다: 망가진 prompt 하나가 무사한 100개보다 중요합니다.

## 벤치마크 패널

임시 prompt는 도구가 작동하는지 알려주고, 벤치마크는 모델이 아직 작동하는지 알려줍니다.

```bash
python scripts/fetch_benchmarks.py        # 파일마다 provenance 기록
python examples/gpt2_bench.py --edits 8
```

| 패널 | 데이터셋 | 라이선스 |
|---|---|---|
| 편집 유효성 | zsRE development | [벤치마크](../reference/benchmarks.md) 참고 |
| Perplexity | WikiText-2 raw test | CC-BY-SA-3.0 AND GFDL |
| 능력 | MMLU test | MIT |

## 출력 읽기

```
== zsRE (edits: 8, layer 8, mean rel.delta 0.0114) ==
  rewrite   0.000 -> 0.500
  rephrase  0.000 -> 0.889   (513 held-out paraphrases)

== WikiText-2 (held out, 20 documents) ==
  perplexity ratio  1.0806

== MMLU (100 questions) ==
  accuracy  0.220 -> 0.220   (delta +0.000)
```

- 유효성은 실제: rewrite 8개 중 4개, 바꿔쓰기 513개 중 456개가 뒤집힘.
- 비용도 실제: perplexity 8% 상승.
- GPT-2의 MMLU는 거친 신호입니다. 1.5B+ 패널이 더 미세한 차이를 해상합니다. 보고하되 과해석하지 마세요.

## 장문 일관성

단답 지표는 가장 흔한 실패를 놓칩니다: 여전히 정답이지만 생성이 무너지는 경우.

```python
from machaon.hygieia import LONGFORM_GATES, longform_report
from machaon.hygieia.gates import evaluate

report = longform_report(pristine, edited, prompts, max_new_tokens=120)
evaluate(report, LONGFORM_GATES)
```

| 지표 | 정의 |
|---|---|
| `longform_nll_ratio` | 편집 모델 텍스트를 **원본 모델**로 평가한 NLL / baseline NLL |
| `longform_repetition_delta` | 편집 후 4-gram 반복률 − baseline |
| `longform_empty_rate` | 빈 생성 비율 |

judge model도 추가 다운로드도 필요 없습니다. 국소 편집이라면 ratio는 1.0 근처, 반복 delta는 0.0 근처에 머물아야 합니다.

## 우리가 지키는 규칙

1. base model이 **틀리는** 항목만 편집해 개선을 선택 편향으로 만들지 않는다.
2. base model이 이미 정답을 내는 바꿔쓰기 집합을 쓴다.
3. 모든 숫자 옆에 provenance(데이터셋, revision, 라이선스, SHA-256)를 기록한다.
4. 개선되지 않은 부분도 보고한다.
