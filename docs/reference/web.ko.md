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
machaon-web --allowed-root /models --port 8765
```

실행하면 일회용 token이 포함된 로컬 URL을 출력합니다. 서버는 기본적으로
`127.0.0.1`에만 바인딩됩니다. `--allowed-root` 또는
`MACHAON_WEB_ALLOWED_ROOTS`로 checkpoint 루트를 지정하세요.

## 표시 언어

콘솔은 English, 중국어 간체, 日本語, 한국어를 제공합니다. 상단 언어 선택기는
정적 문구뿐 아니라 작업 상태, 지표, 오류 메시지도 전환하며 선택을
`localStorage`에 저장합니다. `?lang=zh`, `?lang=ja`, `?lang=ko` 직접 링크도
지원합니다. 명시 선택이 없으면 브라우저 언어를 따르고, 지원하지 않으면 English로
폴백합니다.

## 세 열 작업 흐름

1. **Checkpoint**: safetensors 파일 또는 분할 디렉터리의 header를 읽어
   tensor 수, 파라미터 수, dtype, shard 수를 보여 줍니다. 브라우저는 가중치
   값을 받지 않습니다.
2. **편집 계약**: session을 열고 baseline을 캡처한 뒤 layer를 골라 proposal을
   계산합니다. preserve prompts는 영공간 제약이 되며, 양자화 모델의 gradient
   solve에서는 **Densify from edit layer**를 켭니다. proposal은 쓰기 전에
   상대 변화량과 target probability를 보고합니다.
3. **증거**: proposal을 apply하고 baseline으로 evaluate합니다. fail-closed
   gate 판정과 이벤트 타임라인을 확인하고, rollback으로 원래 tensor 바이트를
   복원합니다.
   저장할 때는 Applied session의 delta를 새 checkpoint 디렉터리에 쓰며 원본은
   변경하지 않습니다.

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
