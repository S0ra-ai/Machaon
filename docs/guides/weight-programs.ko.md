# 가중치 프로그램

가중치 프로그램은 완전하고 검토 가능한 가중치 공간 작업을 설명하는 작은 텍스트 파일입니다.

- 어느 체크포인트에서 시작할지;
- 모델이 어떤 사실을 배워야 하는지;
- 어떤 프롬프트가 변하면 안 되는지;
- 모델을 얼마나 바꿀 수 있는지;
- 저장 후 다시 로드해 검증할지.

`machaon first`, `machaon teach-many`, 브라우저 콘솔이 쓰는 같은 루프를 공유 가능한 형태로 만든 것입니다.
프로그램은 두 번째 편집 엔진을 추가하지 않습니다. 경로와 정책을 해석하고, 티칭 체인을 호출한 뒤,
실제로 기록된 바이트를 검사합니다.

## 완전한 프로그램

`hello.machaon`으로 저장합니다.

```text
version: 1
name: Hello, weight space

model: ./models/SmolLM2-135M
fetch: smol135m
output: ./models/hello-weight-space

budget: 0.10
max_rel_norm: 0.02
verify: true

teach: The Eiffel Tower is in the city of -> Rome
teach: The capital of France is -> Paris

protect: The capital of Germany is
protect: Water freezes at a temperature of
```

모델을 다운로드하거나 로드하지 않고 해석된 계약을 확인합니다.

```bash
machaon program hello.machaon --plan
```

실행합니다.

```bash
machaon program hello.machaon
# 감사 기록을 체크포인트 디렉터리 밖에 두려면:
machaon program hello.machaon --report-out ./reports/hello-progress.json
```

`model`이 없고 `fetch`가 starter를 지정한 경우에만 다운로드합니다. 각 사실은 독립적으로
검색되고 국소 검증되며, 체인은 마지막으로 전역 안전한 체크포인트를 `<output>/final`에 저장합니다.
`report.json`과 `<output>/program-report.json`도 기록됩니다.

`verify: true`는 저장 후 체크포인트를 다시 로드해 모든 사실을 재검사합니다.
저장은 되었지만 재검증에 실패하면 상태가 `verification_failed`가 되고 종료 코드는 0이 아닙니다.
바이트를 별도로 검증할 때만 `--no-verify`를 사용하세요.

로컬 Web 콘솔의 **가중치 프로그램 실행** 패널도 같은 계약을 사용합니다.
프로그램을 붙여 넣고 상대 경로 기준 폴더를 지정한 뒤 **프로그램 확인** 또는
**프로그램 실행**을 선택하세요. 브라우저 경로도 기존 파일 허용 목록을 통과합니다.
패널에서 저장된 보고서를 열어 `machaon program-verify`와 같은 무결성 검사 또는
저장 바이트 검증을 실행할 수도 있습니다.

## 필드

필수는 `model`, `output`, 하나 이상의 `teach`뿐입니다.

| 필드 | 기본값 | 의미 |
|---|---|---|
| `model` | 필수 | 체크포인트 디렉터리. 상대 경로는 프로그램 파일 기준 |
| `output` | 필수 | 새 체크포인트 컨테이너. 비어 있지 않은 기존 디렉터리는 거부 |
| `fetch` | 없음 | `model`이 없을 때 받을 starter: `smol135m`, `qwen05b`, `gpt2` |
| `teach` | 필수, 반복 가능 | 티칭 문장 또는 JSON `prompt` / `answer` 객체 |
| `protect` | 없음, 반복 가능 | 기준선, 투영, 패널에 쓰는 무관한 프롬프트 |
| `budget` | `0.10` | 모든 사실의 누적 상대 노름 예산 |
| `max_rel_norm` | `0.02` | 사실별 상대 노름 상한 |
| `max_kl` | `0.05` | 보호 패널의 KL p95 상한 |
| `min_panel_agreement` | `0.875` | top-1을 유지해야 하는 보호 프롬프트 비율 |
| `layers` | 자동 깊이 창 | `18,20,22` 또는 `18-22` |
| `solver` | `direction` | `direction` 또는 `gradient` |
| `dtype` | `float32` | `auto`, `float32`, `float16`, `bfloat16` |
| `quantize` | 없음 | `int4` 또는 `int8`; `dtype: auto` 필요 |
| `device` | 런타임 기본값 | `cpu`, `mps`, `cuda` 또는 런타임 값 |
| `verify` | `true` | 기록된 체크포인트를 다시 로드해 모든 사실 검사 |
| `continue_on_reject` | `false` | 거부된 사실을 기록하고 계속 |
| `keep_steps` | `false` | 안전한 final뿐 아니라 각 단계도 보존 |

`#`은 주석이고 빈 줄은 무시됩니다. `key: value` 또는 `key = value`를 쓰며,
`teach`와 `protect`는 여러 번 나올 수 있습니다.

## 상대 경로

`model`과 `output`의 상대 경로는 프로그램 파일이 있는 디렉터리 기준으로 해석됩니다.
프로그램과 `models/` 폴더를 다른 컴퓨터로 복사해도 계약이 바뀌지 않게 하기 위해서입니다.

`--out DIR`은 해당 실행의 출력만 덮어씁니다. 프로그램 해시와 원래 선언된 출력은
보고서에 남으므로 공유된 계약을 몰래 바꿀 수 없습니다.

```bash
machaon program hello.machaon --out ./models/experiment-2
```

## JSON 형식

같은 계약을 JSON으로 생성할 수 있습니다. 사실은 문자열, `{"say": "..."}`,
또는 명시적 `{"prompt": "...", "answer": "..."}` 객체입니다.

```json
{
  "version": 1,
  "model": "./models/SmolLM2-135M",
  "fetch": "smol135m",
  "output": "./models/hello-weight-space",
  "facts": [
    "The Eiffel Tower is in the city of -> Rome",
    {"prompt": "The capital of France is", "answer": " Paris"}
  ],
  "protect": ["Two plus two equals"],
  "verify": true
}
```

두 형식 모두 같은 `program-report.json`을 만들며, 프로그램 SHA-256, 해석된 정책,
각 티칭 단계, 검증 결과를 포함합니다. 공유하거나 검토할 때의 감사 기록입니다.

## 보고서를 신뢰하지 않고 검증하기

저장된 보고서는 주소 필드를 제외한 정규화 문서로 계산한 `report_sha256`을 함께 담습니다.
따라서 복사하거나 커밋하고, 이슈에 첨부하거나 브라우저에서 내려받은 뒤에도 검증할 수 있습니다.

```bash
machaon program-verify ./models/hello-weight-space/program-report.json
machaon program-verify ./models/hello-weight-space/program-report.json \
  --program hello.machaon
machaon program-verify ./models/hello-weight-space/program-report.json \
  --check-model
```

첫 번째 명령은 보고서 주소와 필수 구조를 검증합니다. 두 번째는 프로그램 파일의 해시가
보고서에 기록된 프로그램 주소와 같은지도 요구합니다. 세 번째는 보고서가 가리키는 저장된
체크포인트를 다시 불러와 모든 사실을 재실행하고, 실측 결과가 기록된 판정과 일치하는지
보여 줍니다.

체크포인트를 옮겼다면 새 위치를 명시할 수 있습니다.

```bash
machaon program-verify ./reports/lesson.json \
  --check-model --checkpoint ./copied/lesson/final
```

SHA-256은 내용 주소이며 서명이 아닙니다. 우발적이거나 승인되지 않은 변경을 탐지하지만,
바이트를 바꾼 뒤 누구나 주소를 다시 계산할 수 있습니다. `--check-model`은 선언한 사실을
저장된 가중치로 실제 실행하므로 더 강한 검사입니다. 주소 필드가 없는 이전 보고서는
검증된 것으로 취급하지 않고 거부합니다.

## 주장하지 않는 것

국소 검사는 선언한 사실이 성립하고 보호 패널이 범위 안에 머물렀음을 보여줍니다.
벤치마크를 대체하지 않습니다. 프로그램 보고서는 이 편집에 대한 증거이며,
모델 전체에 대한 일반적 결론이 아닙니다.
