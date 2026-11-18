# 웹 콘솔

로컬 웹 콘솔은 Python API, CLI, MCP 서버와 동일한
`machaon.tools` / `machaon.session` 위에 놓인 얇은 진입점입니다. 모델 로직을
복제하지 않고, 세 진입점이 같은 트랜잭션 계약을 공유합니다.

```text
브라우저 ──> FastAPI / SSE ──> machaon.tools / machaon.session
                                  │
                    CLI ──────────┤
                    MCP server ───┘
```

## 설치와 실행

핵심 패키지는 NumPy만 필요합니다. 웹 콘솔을 사용할 때 선택 의존성을 추가합니다.

```bash
pip install "machaon[web]"
machaon web --allowed-root /models --port 8765
# machaon-web도 같은 실행기입니다
```

실행하면 일회용 token이 포함된 로컬 URL을 출력합니다. 서버는 기본적으로
`127.0.0.1`에만 바인딩됩니다. `--allowed-root` 또는
`MACHAON_WEB_ALLOWED_ROOTS`로 checkpoint 루트를 지정하세요.
`torch`, `transformers`, `safetensors`가 없으면 첫 session이 조용히 실패하는 대신
정확한 `pip install "machaon[hf]"` 명령을 banner로 보여 줍니다.

## 표시 언어

콘솔은 English, 중국어 간체, 日本語, 한국어를 제공합니다. 상단 언어 선택기는
정적 문구뿐 아니라 작업 상태, 지표, 오류 메시지도 전환하며 선택을
`localStorage`에 저장합니다. `?lang=zh`, `?lang=ja`, `?lang=ko` 직접 링크도
지원합니다. 명시 선택이 없으면 브라우저 언어를 따르고, 지원하지 않으면 English로
폴백합니다.

## 첫 방문 가이드

처음 인증하면 **FIRST EDIT** 0에서 시작 카드를 먼저 보여 주고, 편집의 목적,
검사 → 세션 열기 → baseline → 제안 → 적용 → 평가 →
저장/롤백 흐름, 각 컨트롤의 의미를 설명하는 짧은 가이드가 열립니다.
**탐색 시작**을 누르면 읽음으로 기록되고, 상단의 **가이드**로 언제든 다시 열 수
있습니다. Hero 아래의 4단계 안내는 가벼운 리마인더로 남겨 둡니다.

## 첫 편집 카드

첫 인증 화면에는 **FIRST EDIT** 카드도 표시됩니다. 체크포인트를 선택하지 않았다면
카드에서 스타터를 고를 수 있습니다(기본 SmolLM2-135M, 다국어용
Qwen2.5-0.5B, 또는 GPT-2). 선택한 모델을 받아 열고,
"The Eiffel Tower is in the city of -> Rome"을 미리 채운
뒤 교수 레시피를 실행해 결과를 보여 줍니다. 두 예시 버튼은 편집이 적용되는
`Eiffel → Rome`과 모델이 이미 아는 `France → Paris`를 불러오므로, 입력하지 않고도
두 결과를 볼 수 있습니다. session이 열리면 카드는 사라지고,
수용된 편집은 **저장**을 위해 session에 남습니다. 교수 적용 후 카드에는
선택된 레이어, 상대 Δ, 패널 보존 수, KL p95가 표시되어 첫 성공도 고급
워크스페이스와 같은 증거 용어를 갖습니다. 또한 `<model>-machaon`이 채워진
**체크포인트 출력** 필드와 **체크포인트 저장** 버튼이
남으며, 오른쪽 열과 같은 `/api/session/save`를 사용합니다. 저장 후
**저장된 체크포인트 열기**는 편집된 session을 닫고(메모리 바이트 롤백) 저장된
체크포인트를 열어 다음 사실을 위한 깨끗한 session을 남깁니다. **완료**는 현재
session을 유지합니다. 연 뒤에는 먼저 방금 가르친 prompt/answer를 다시 연
체크포인트에서 자동 평가해, 저장된 답변이 재현되는지 보여 줍니다. 그다음 문장을
비우고 출력을 새 체크포인트의 다음 번호 `<name>-machaon-N`(예:
`<name>-machaon-2`)으로 갱신하며 **다른 사실 가르치기**를 제공해, 메인 폼으로
돌아가지 않고 같은 루프를 계속할 수 있습니다.
저장할 때마다 새 session과 기준선이
시작되므로, 사실 간 누적 예산과 원래 panel 드리프트 검사에는 **여러 사실
가르치기**를 사용하세요. **숨기기**는 `localStorage`에
저장됩니다.

마지막으로 정상적으로 열거나 저장한 체크포인트 경로도 `localStorage`에 기록됩니다.
새로고침하면 Checkpoint 입력란에 복원되어, 의도치 않게 다른 starter를 다시
다운로드하지 않습니다. FIRST EDIT 상태에도 사용할 체크포인트가 표시됩니다.

저장 후 카드의 **여러 사실 가르치기**는 다사실 패널을 열고 현재 사실을 미리 채우며
`<model>-machaon-many` 출력 컨테이너를 제안합니다. 목록 결과는 상태 칩뿐 아니라
각 사실의 prompt, 상태, 레이어, Δ, 전역 드리프트를 펼쳐 보여 줍니다.

## 저장된 사실 확인

Checkpoint 열의 **저장된 사실 확인** 패널은 session을 열거나 수정하지 않고 선택한
체크포인트의 prompt/answer 또는 한 줄당 하나의 사실 목록을 평가합니다.
`/api/check`를 호출해 저장된 가중치를 다시 읽고, `machaon check`와 같은 앞 공백
처리를 사용해 각 사실의 PASS/FAIL, 실제 답변, 정확도를 표시합니다.

## 가중치 프로그램 실행

Checkpoint 열의 **가중치 프로그램 실행** 패널은 `machaon program`과 같은
`.machaon` 텍스트 또는 JSON을 받습니다. **프로그램 폴더**를 허용 루트 아래
디렉터리로 지정하면 프로그램의 상대 `model` / `output` 경로가 그곳에서
해석됩니다. **프로그램 확인**은 `/api/program/plan`을 호출해 모델을 읽지 않고
모델, 사실, 예산, 검증 정책을 보여 줍니다. 카드는 출력 경로가 새 경로인지, 비어 있고
사용 가능한지, 이미 막혔는지도 보여 주며 모델이 없으면 starter와 다운로드 크기를
표시합니다. **프로그램 실행**은 백그라운드
작업으로 `/api/program/run`을 호출하고 `<output>/program-report.json`을 쓴 뒤,
저장된 체크포인트를 다시 읽어 검증한 다음 보고합니다. 작업 배너는 각 사실의
step/total과 상태를, 마지막에는 결과와 저장 바이트 검증을 표시합니다.
선택적 **보고서 경로**는 같은 보고서를 `<output>/program-report.json` 이외의 위치에
씁니다. 실행 카드에는 보고서의 SHA-256이 표시됩니다.

**보고서 검증**은 이전에 내려받은 `program-report.json`을 열어 내용 주소와 구조,
기록된 프로그램 주소, 상태, 판정을 확인합니다. **보고서 + 체크포인트 검증**은
체크포인트를 다시 불러와 모든 사실을 재실행하고, 실측 결과가 보고서와 일치하는지
보여 줍니다. **편집기의 프로그램도 함께 일치 확인**을 선택하면 다른 프로그램 파일을
탐지할 수 있고, 저장 바이트를 옮겼다면 **현재 선택한 체크포인트를 다시 검증**을
사용합니다. **이번 실행의 보고서 검증**은 페이지에 이미 있는 결과를 직접 검사하므로
다운로드하고 다시 선택할 필요가 없습니다. 이 검사는 `/api/program/verify` 백그라운드
작업으로 실행되므로 큰
체크포인트를 읽는 동안에도 페이지가 멈추지 않습니다.

**현재 체크포인트 사용**은 열린 session 또는 선택한 체크포인트, 교수 문장,
preserve prompts로 편집기를 채웁니다. **.machaon 다운로드**는 편집기 내용을
이식 가능한 파일로 저장하므로 브라우저에서 성공한 레시피를 그대로 공유할 수 있습니다.
**.machaon 열기**는 로컬 텍스트/JSON 파일을 편집기로 읽어 오며, 확인·실행 시에는
동일한 허용 루트 제약이 적용됩니다.
세 내장 템플릿은 기존 체크포인트가 없어도 됩니다. **SmolLM2 첫 편집**은 한 사실,
**두 가지 사실**은 두 사실 체인, **Qwen 중영 이중언어**는 Qwen2.5-0.5B에서 영어와
중국어를 같은 프로그램으로 다룹니다.
**프로그램 확인** 또는 실행을 시작한 최근 8개 프로그램은 브라우저 로컬 기록에 남고,
**불러오기**로 복원할 수 있습니다. 업로드하지 않으며 `localStorage`에만 저장합니다.
**프로그램 복사**는 편집기 내용을, **실행 보고서 다운로드**는 현재 실행 JSON을 저장하며,
**출력 체크포인트 사용**은 final을 새 session에서 바로 엽니다.
**사실로 프로그램 만들기**는 한 줄에 하나의 교수 문장을 현재 체크포인트, 동일한 출력
이름 규칙, 기본 예산과 검증 정책을 사용하는 검토 가능한 `.machaon`으로 변환합니다.
체크포인트가 없으면 선택한 SmolLM2/Qwen/GPT-2 starter와 해당 `fetch` 지시문을 자동으로 넣습니다.
이 버튼은 `/api/program/new`를 호출하며 `machaon program-new`와 같은 생성기와 파서를
공유합니다. 편집 열의 preserve prompts는 `protect` 항목이 됩니다.
완료된 실행은 **최근 실행**에 추가되며, **출력 열기**로 final 체크포인트를 경로 입력 없이 다시 열 수 있습니다.

두 경로 모두 다른 콘솔 작업과 같은 파일 시스템 허용 목록을 적용합니다.
붙여 넣은 프로그램만으로 허용 루트 밖을 읽거나 쓸 수는 없습니다.

## 세 열 작업 흐름

1. **Checkpoint**: safetensors 파일 또는 분할 디렉터리의 header를 읽어
   tensor 수, 파라미터 수, dtype, shard 수를 보여 줍니다. 브라우저는 가중치
   값을 받지 않습니다. 체크포인트가 없으면 starter를 고르고
   **Download starter**를 누르세요. 허용 루트로 미러를 포함해 다운로드하고
   경로를 채웁니다. 작업 배너는 SSE로 현재 엔드포인트, 파일, 바이트/퍼센트
   진행률을 보여 줍니다. **다운로드 후 열기**를 켜면 session까지 자동으로 열립니다.
2. **편집 계약**: session을 열고 baseline을 캡처한 뒤 layer를 골라 proposal을
   계산합니다. preserve prompts는 영공간 제약이 되며, 양자화 모델의 gradient
   solve에서는 **Densify from edit layer**를 켭니다. proposal은 쓰기 전에
   상대 변화량과 target probability를 보고합니다.
3. **증거**: proposal을 apply하고 baseline으로 evaluate합니다. fail-closed
   gate 판정과 이벤트 타임라인을 확인하고, rollback으로 원래 tensor 바이트를
   복원합니다.
   저장할 때는 Applied session의 delta를 새 checkpoint 디렉터리에 쓰며 원본은
   변경하지 않습니다.

### 한 문장으로 가르치기

편집 열 상단의 **ONE SENTENCE** 상자가 가장 짧은 경로입니다. `machaon intent`와
같은 문법을 받아들이고 `machaon teach`와 같은 교수 레시피를 실행합니다.

1. **Preview parse**는 문장을 컴파일해 prompt/answer 필드에 채웁니다.
2. **Teach automatically**는 깊이 창을 검색하고, preserve prompts의 영공간으로
   편집을 투영한 뒤 최선의 후보를 적용합니다. 목표, 예산, panel, KL 검사가 모두
   통과할 때만 유지하며 거부된 후보는 자동으로 롤백합니다.

선택적인 **Also write the edited checkpoint** 필드에 경로를 쓰면 허용 루트 아래의
새 디렉터리에 저장합니다. 비워 두면 session에 유지되고 나중에 **Save**로 저장할
수 있습니다.

**여러 사실 가르치기**도 같은 상자에서 제공됩니다. 한 줄에 하나의 문장을
받아 사실별 검색을 실행하고 `<out>/final`과 `<out>/report.json`을 출력
컨테이너에 기록합니다. 현재 session은 base에 남으며 **최종 체크포인트 사용**으로
마지막 전역 안전 결과를 열 수 있습니다.

모델 로딩, solve, 평가는 단일 worker에서 실행되며 SSE로 진행 상황을 전송합니다.
프로세스마다 하나의 모델 session을 유지합니다.

적용 후 평가가 끝나면 **Persist**로 현재 delta를 새 checkpoint에 기록합니다.
단일 tensor는 `Checkpoint.write_tensor`, 여러 tensor는
`Checkpoint.write_with`를 사용합니다. 원본 checkpoint는 바뀌지 않으며 출력
디렉터리는 허용 루트 안에 있어야 합니다.

## 대형 체크포인트 예시: Qwen3.6-27B

콘솔은 모델에 독립적이며 같은 컨트롤이 모든 지원 아키텍처에 적용됩니다.
27B는 대형 체크포인트의 한 예시입니다. 34 GB 통합 메모리 환경에서는 `int4`
또는 `int8`을 사용하고 언어 디코더의 뒤쪽 layer를 편집 지점으로 삼습니다.

```text
path:      ./qwen36-27b
dtype:     auto
quantize:  int4
device:    cpu
layer:     60
budget β:  0.01
module:    mlp.down_proj
solver:    gradient
densify:   on
```

먼저 header만으로 checkpoint를 확인하세요. Transformers runtime, 단일 shard
쓰기, `Checkpoint.write_tensor`의 자세한 내용은
[Qwen3.6-27B 가이드](../guides/qwen27b.md)에 있습니다.

## 운영 규칙

콘솔은 loopback에 두고, 허용 루트를 실제로 편집하는 checkpoint 디렉터리로
좁히세요. 출력된 token은 session과 checkpoint 쓰기를 승인하는 임시 capability
입니다. 웹 계층은 임의 Python 코드를 받지 않으며 경로 허용 목록, 예산 검사,
트랜잭션 rollback을 통과합니다.
