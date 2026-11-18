# CLI

```bash
python -m machaon <command>
```

종료 코드는 계약의 일부이며 CI 단계로 바로 쓸 수 있습니다:

| 종료 코드 | 의미 |
|---|---|
| 0 | 성공 / gate 통과 |
| 1 | gate가 편집을 거부, 또는 검증 실패 |
| 2 | 입력 형식 오류, 파일 없음, 지원하지 않는 체크포인트 |

## `first`

0에서 시작하는 최단 경로입니다. `--model`이 없으면 `machaon fetch`와 같은
엔드포인트 폴백과 헤더 검증으로 선택한 starter(기본 SmolLM2-135M)를 받고,
체크포인트를 연 뒤 한 가지 사실을 가르치고 수용된 편집을 저장합니다.

```bash
machaon first
machaon first --say "프랑스의 수도는 -> 베를린"
machaon first --model ./models/Qwen2.5-0.5B --out ./models/qwen-first
```

기본 사실은 "The Eiffel Tower is in the city of -> Rome", 기본 출력은
`<model>-machaon`입니다. `--no-save`는 메모리에만 유지하고 `--json`은 전체
결과를 출력합니다. 적용/이미 알고 있으면 0, 로컬 검사 거부는 1, 입력 또는
모델 오류는 2로 종료합니다.

## `check`

저장된 체크포인트를 다시 읽고 가중치를 바꾸지 않은 채 prompt/answer 한 쌍만
재검증합니다. 답변의 앞 공백 유무를 모두 시도해 모델의 next-token 예측과 맞는
버전을 보고하며, 목표 token이 재현될 때만 종료 코드 0입니다.

```bash
machaon check --model ./models/qwen-first \
  --say "The Eiffel Tower is in the city of -> Rome"
machaon check --model ./models/qwen-first \
  --prompt "The Eiffel Tower is in the city of" --answer Rome --json
machaon check --model ./models/qwen-many/final --facts facts.txt
```

기본 `--dtype auto`는 저장된 dtype을 읽습니다. `--quantize`에는
`--dtype auto`가 필요합니다. JSON은 `passed`, `accuracy`, `answer_text`와
원본 metrics를 포함하며, 사람이 읽는 출력은 `PASS` / `FAIL`입니다. `--facts`는
체크포인트를 한 번만 읽고 각 사실을 검사하며, 하나라도 실패하면 종료 코드 1입니다.

## `fetch`

torch를 가져오지 않고 가중치 tensor도 펼치지 않은 채 safetensors 체크포인트를
다운로드합니다. Hugging Face, `hf-mirror`, ModelScope 순으로 저장소를 나열하고
각 `.safetensors` 헤더를 검증한 뒤 받아들입니다. 기존 파일은 재사용하므로
중단된 다운로드를 이어갈 수 있습니다.

```bash
machaon fetch --list
machaon fetch --starter qwen05b --dest ./models/Qwen2.5-0.5B
machaon fetch --starter smol135m --dest ./models/SmolLM2-135M
machaon fetch --repo Qwen/Qwen2.5-0.5B --dest ./models/qwen
```

`--endpoint huggingface|hf-mirror|modelscope`로 하나의 엔드포인트를 강제할 수
있습니다. `--json`은 파일 목록, 샤드 수, tensor 수, 파라미터 수를 출력합니다.
`.safetensors` 가중치가 없는 저장소는 반쯤 받은 디렉터리를 남기지 않고 거부합니다.

## `start`

첫 편집까지의 최단 경로와 설치된 선택 런타임을 출력합니다.

```bash
python -m machaon.cli start
python -m machaon.cli start --demo
```

## `web`

메인 CLI에서 선택적 로컬 브라우저 콘솔을 실행합니다. host, port, 허용 루트,
token, loopback 보호는 `machaon-web`과 같습니다. 먼저 `web` extra를 설치하세요.

```bash
pip install "machaon[web]"
machaon web --allowed-root ./models
machaon web --allowed-root ./models --port 8770 --no-browser
```

## `intent`

자연어 교수 요청을 모델을 읽지 않고 정확한 `prompt` / `answer` 계약으로
컴파일합니다. 종료 코드 2는 한 쌍으로 확정하지 못했다는 뜻이며, 오류에
지원하는 형식이 표시됩니다.

```bash
python -m machaon.cli intent "에펠탑은 -> 로마"
python -m machaon.cli intent "教会模型：北京的首都是 → 北京" --json
```

화살표와 라벨 형식은 정확합니다. 일반 문장도 받지만 분할 지점이 가정이므로
`medium` 신뢰도로 표시됩니다. 동등한 `machaon teach` 명령도 출력합니다.

## `teach`

기준선, 깊이 방향 layer 검색, 적용, 목표/panel/KL 검사, 실패 시 롤백까지
한 명령으로 실행합니다.

```bash
python -m machaon.cli teach --model ./models/Qwen2.5-0.5B \
  --say "에펠탑은 -> 로마"
```

기본 출력은 `<checkpoint>-machaon`이라는 형제 디렉터리이며 원본
체크포인트는 변경되지 않습니다. 주요 옵션:

| 플래그 | 의미 |
|---|---|
| `--say TEXT` | `machaon intent`로 문장을 컴파일 |
| `--prompt P --answer A` | 계약을 명시 |
| `--layers 8,10,12` / `--layers 8-12` | 후보 layer 지정 |
| `--max-layers 7` | 기본 깊이 창의 후보 수 |
| `--solver direction\|gradient` | 각 후보의 value solver |
| `--max-rel-norm 0.02` | 편집당 상대 노름 예산 |
| `--budget 0.10` | 세션 누적 예산 |
| `--max-kl 0.05` | KL p95 승인 임계값 |
| `--min-panel-agreement 0.875` | 무관 prompt의 top-1 유지율 하한 |
| `--preserve-file FILE` | 한 줄에 하나씩 기준선/투영 prompt |
| `--corpus-file FILE` | 선택적 perplexity 검사용 텍스트 |
| `--out DIR` | 편집된 체크포인트(이미 있으면 거부) |
| `--no-save` | 메모리에만 유지 |
| `--dry-run` | 검색과 보고만, 쓰기 없음 |
| `--evidence-out DIR` | 편집 전 headroom이 포함된 WSP 증거 저장 |
| `--edit-out DIR` | 적용된 delta를 OpenWSP v1로 내보내기 |
| `--report-out FILE` | 전체 레시피 보고서 JSON 저장 |
| `--json` | 기계 판독 출력 |

종료 코드: `0` 적용 / 변경 없음 / 드라이런, `1` 거부 후 롤백,
`2` 입력 또는 모델 오류.

`--out`, `--evidence-out`, `--edit-out` 대상이 이미 있으면 이전 산출물을 덮어쓰지 않도록 거부합니다.

## `teach-many`

여러 사실을 하나의 체크포인트 체인으로 가르칩니다. 각 사실은 직전에 수용된
체크포인트에서 새로 연 깨끗한 session에서 단일 사실 레시피 전체를 실행합니다.
원본 체크포인트는 변경되지 않습니다.

```bash
machaon teach-many --model ./models/Qwen2.5-0.5B \
  --facts facts.txt --out ./models/qwen-many
```

`facts.txt`는 한 줄에 하나의 교수 문장(`machaon intent`와 같은 문법)이거나
JSON / JSONL의 `{"say": ...}` 또는 `{"prompt": ..., "answer": ...}` 목록입니다.
마지막으로 전역 안전한 체크포인트를 `<out>/final`에 두고 각 단계 증거를
`<out>/report.json`에 씁니다. 사실이 거부되거나 누적 드리프트 검사가 실패하면
종료 코드 1입니다. `--continue-on-reject`는 실패를 기록하고 계속하며,
`--keep-steps`는 마지막 안전 단계뿐 아니라 모든 단계를 보존합니다. 이 경우
final은 `<out>/final`이 아니라 마지막 `step-NNN` 디렉터리입니다. 누적
`--budget`은 사실별이 아니라 체인 전체에 적용됩니다.

## `program-new`

하나 이상의 교수 문장에서 검토 가능한 텍스트 프로그램을 생성합니다. 생성 파일은
`machaon program`과 같은 검증기로 파싱되므로 두 번째 형식을 만들지 않습니다.

```bash
machaon program-new ./hello.machaon \
  --teach "The Eiffel Tower is in the city of -> Rome" \
  --protect "Two plus two equals"

machaon program-new - \
  --model ./models/Qwen2.5-0.5B \
  --teach "The capital of France is -> Paris"
```

기본값은 `--model ./models/SmolLM2-135M`, `fetch: smol135m`,
`--output <model>-machaon`입니다. 사용자 지정 `--model`에는 암시적 `fetch`가 없습니다.
필요하면 `--fetch smol135m|qwen05b|gpt2` 또는 `--no-fetch`를 명시하세요.
`--teach`와 `--protect`는 반복할 수 있고, `--budget`과 `--max-rel-norm`은 자주 쓰는
정책 값입니다. 경로 `-`는 표준 출력으로 쓰며, 기존 파일은 `--force` 없이 거부됩니다.

## `program`

이식 가능한 가중치 프로그램을 계획하거나 실행합니다. 프로그램 파일(텍스트 또는
JSON)은 모델, 사실, 보호 프롬프트, 예산, 런타임 정책을 선언합니다. 상대 모델/출력
경로는 프로그램 파일 기준으로 해석되므로 같은 파일을 공유하고 검토할 수 있습니다.

```bash
machaon program examples/hello.machaon --plan
machaon program examples/hello.machaon
machaon program ./lesson.json --out ./models/experiment-2
```

`--plan`은 모델을 다운로드하거나 로드하지 않고 해석된 계약만 출력합니다. 실행은
편집을 `teach-many`에 맡기고 `<output>/program-report.json`을 쓰며, 기본적으로
저장된 체크포인트를 다시 로드해 모든 사실을 재검증합니다. `--no-verify`는 마지막
재검증을 생략하고, `--json`은 계획 또는 실행 결과 전체를 출력합니다. 사실별
진행은 기본으로 출력되며 `--quiet`로 끌 수 있습니다. `--report-out PATH`는
프로그램 보고서를 `<output>/program-report.json` 대신 지정한 파일에 씁니다.

종료 코드는 `0`이 유효한 계획 또는 검증된 실행, `1`이 거부/검증 실패, `2`가
프로그램 형식 또는 모델 오류입니다. 형식과 필드 표는
[가중치 프로그램](../guides/weight-programs.md)을 참고하세요.

## `program-verify`

저장된 `program-report.json`을 신뢰하지 않고 검증합니다. 기본 검사는 보고서를 읽고
필수 구조를 확인한 뒤 `report_sha256` 내용 주소를 다시 계산합니다.

```bash
machaon program-verify ./models/hello-weight-space/program-report.json
machaon program-verify ./models/hello-weight-space/program-report.json \
  --program hello.machaon
machaon program-verify ./models/hello-weight-space/program-report.json \
  --check-model
```

`--program PATH`는 프로그램 파일의 SHA-256이 보고서에 기록된 프로그램 주소와 같을 것도
요구합니다. `--check-model`은 보고서가 가리키는 체크포인트를 다시 불러와 선언된 모든
사실을 재실행하고, 실측 판정을 기록된 판정과 비교합니다. 체크포인트를 옮겼다면
`--check-model --checkpoint ./new/path/final`을 사용하세요. 보고서의 경로는 출처이며
현재 위치의 필수 조건은 아닙니다.

`--json`은 전체 검증 결과를 출력합니다. 종료 코드 `0`은 보고서가 온전하고 요청한 모델
검사가 통과했음, `1`은 요청한 저장 바이트 검사 실패, `2`는 보고서 형식 또는 주소가
맞지 않음을 뜻합니다. SHA-256은 변경 탐지이며 서명이 아닙니다.

## `demo`

합성 rank-one 데모 실행. 모델도 네트워크도 필요 없음.

```bash
python -m machaon.cli demo
```

## `validate`

edit spec 검증. spec이 잘못되면 종료 코드 1, JSON이 깨지면 2.

```bash
python -m machaon.cli validate examples/edit_spec.json
# OK  rank1_constrained -> model.layers.14.mlp.down_proj (budget 0.020)
```

## `gate`

metrics JSON에 게이트 적용. 종료 코드 1은 커밋하면 안 된다는 뜻이라 CI에 바로 넣을 수 있습니다.

```bash
python -m machaon.cli gate examples/metrics_pass.json    # exit 0
python -m machaon.cli gate examples/metrics_fail.json    # exit 1
```

```
FAIL
  paraphrase_acc = 0.41 violates >= 0.6
  kl_p95 = 0.113 violates <= 0.05
  ...
```

## `inspect`

safetensors 체크포인트의 tensor 목록.

```bash
python -m machaon.cli inspect /models/qwen --suffix down_proj.weight
python -m machaon.cli inspect /models/qwen --layer 3 --limit 5
python -m machaon.cli inspect /models/qwen --layer 3 --json
```

## `doctor`

설치된 능력과 버전을 보고. issue에 붙이기 좋습니다.

```bash
python -m machaon.cli doctor
python -m machaon.cli doctor --json
```

## `--version`

```bash
python -m machaon --version
```

## CI

```bash
./scripts/ci.sh
```

테스트, 데모, spec 검증, 두 게이트 케이스, lint를 실행합니다.
