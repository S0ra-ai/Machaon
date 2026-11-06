# MCP 서버

Machaon은 Model Context Protocol로 파일 수준 작업을 노출합니다. agent는 모델을 로드하지 않고 체크포인트를 검사하고, 다른 사람의 편집을 감사하고, delta를 적용하고, 결과를 게이트할 수 있습니다.

```bash
pip install "machaon[mcp]"
python -m machaon.mcp_server
```

두 세대 SDK를 지원합니다: `mcp` 1.x는 `FastMCP`, 2.x는 `MCPServer`로 이름이 바뀌었습니다. 데코레이터와 `run(transport="stdio")` 표면은 같고, `create_server()`가 설치된 것을 고릅니다.

CI는 서버를 서브프로세스로 띄우고 실제 클라이언트로 도구를 호출합니다(`scripts/mcp_smoke.py`). 전송이 깨지면 빌드가 실패합니다.

## 클라이언트 설정

Codex:

```bash
codex mcp add machaon -- python -m machaon.mcp_server
```

Claude Desktop(`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "machaon": {
      "command": "python",
      "args": ["-m", "machaon.mcp_server"]
    }
  }
}
```

파일 수준 서버는 stdio로 동작하고 모델 상태를 갖지 않아 재시작 비용이 낮습니다. 모델 수준 도구는 명시적으로 연 하나의 `Session`을 공유합니다.

## 파일 수준 도구

| 도구 | 하는 일 |
|---|---|
| `describe_checkpoint(path)` | tensor 수, 파라미터 수, dtype 구성, 샤드 수——헤더만 |
| `list_tensors(path, contains, suffix, layer, limit)` | 이름, 모양, dtype. 값은 반환하지 않음 |
| `audit_edit(base_path, edit_dir, max_rel_norm)` | OpenWSP v1 편집을 base와 대조하고 드리프트 재계산 |
| `apply_delta(base_path, tensor, delta_path, out_dir)` | 하나의 `.npy` delta를 한 tensor에 적용하고 새 체크포인트 작성 |
| `gate(metrics)` | 게이트 적용. 누락 지표는 실패 |

각 도구는 `machaon.tools`의 평범한 함수에 위임하고 테스트는 그 함수를 직접 검증합니다. 전송은 동작을 추가하지 않습니다.

## 모델 수준 도구

하나의 로드된 session에서 동작합니다. `open_model`이 유일한 고비용 호출입니다.

| 도구 | 하는 일 |
|---|---|
| `open_model(path, dtype)` | 모델을 session에 로드 |
| `model_status()` | 파라미터, 레이어, 편집된 tensor, 예산 사용 |
| `capture_baseline(neutral_prompts, corpus_texts)` | **편집 전** 참조 분포 기록 |
| `propose_edit(prompt, target, layer, solver, steps, max_rel_norm)` | 편집을 풀고 쓰지 않은 채 비용 보고 |
| `apply_edit()` | 대기 중 제안을 쓰고 예산을 원자적으로 청구 |
| `evaluate(prompts, targets)` | 정확도, KL, baseline 대비 perplexity |
| `rollback()` | 편집된 모든 tensor 복원 |

의도된 루프는 `machaon.session`과 같습니다:

```
open_model -> capture_baseline -> propose_edit -> apply_edit -> evaluate
                                                        |
                                                    rollback
```

`propose_edit`이 쓰기 전에 상대 delta를 반환하는 것이 핵심입니다. agent는 너무 비싸다고 판단해 적용하지 않을 수 있습니다.

## 오류 형태

Session 도구는 예외 대신 `{"error": "no model is open; call open_model first"}`를 반환해, agent가 일반적인 `Error executing tool` 대신 이유를 봅니다.

## 설계 경계

- **파일 수준은 헤더 우선.** 검사, 감사, 체크포인트 재작성은 모델을 메모리에 올리지 않습니다.
- **모델 수준은 명시적.** `open_model`이 유일한 고비용 호출이고, `propose_edit`은 `apply_edit` 전에 비용을 보고합니다.
- **편집 위치는 agent가 결정.** 레이어 선택은 agent 또는 `machaon.procedures.rome`의 일이며, 서버는 트랜잭션과 예산을 강제합니다.
- **경로는 프로세스 권한 사용.** shell과 같은 신뢰 경계에서 실행하세요. 자세한 내용은 [SECURITY.md](https://github.com/S0ra-ai/Machaon/blob/main/SECURITY.md).
