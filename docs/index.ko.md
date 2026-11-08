# Machaon

**LLM을 위한 직접적인 가중치 수술 — 그래디언트가 아니라 가중치를 편집합니다.**

*화살만 뽑고, 몸은 그대로 둡니다.*

Machaon은 모델 파라미터를 직접 수정한 뒤, 그 변경이 다른 능력을 망가뜨리지 않았음을 증명합니다. 무엇을 건드려도 되는지, 실제로 무엇이 움직였는지, 답이 틀렸을 때 어떻게 되돌리는지까지. 한 번의 편집은 diff하고 재생하고 롤백할 수 있는, 내용 주소 기반의 작은 산출물입니다.

## 세 가지 보장

| 보장 | 구현 | 검증 |
|---|---|---|
| 보호한 key는 움직이지 않는다 | 영공간 투영, `ΔW k⊥ = 0` | `verify.check_protected` |
| 거부된 편집은 정확히 되돌아간다 | 적용 전 원본 바이트 복사 | `transaction.rollback` |
| 측정하지 않은 편집은 통과하지 못한다 | 누락 지표는 실패로 처리 | `hygieia.evaluate` |
| 편집은 재현 가능하다 | delta를 내용 주소로 저장 | `store.DeltaStore` |

## 설치

```bash
git clone https://github.com/S0ra-ai/Machaon.git
cd Machaon

uv pip install -e .              # NumPy만: 수학, 게이트, 상호운용
uv pip install -e ".[hf]"        # + torch + transformers, 실제 모델
uv pip install -e ".[bench]"     # + pyarrow, 벤치마크
uv pip install -e ".[mcp]"       # + MCP SDK, Python 3.10+
uv pip install -e ".[web]"       # + FastAPI + uvicorn, 로컬 브라우저 콘솔
uv pip install -e ".[dev,docs]"  # 테스트, lint, 커버리지, 문서
```

Python 3.9–3.13을 테스트합니다. `make ci`로 오프라인 검사를 한 번에 실행할 수 있습니다.

## 30초 체험

```python
import numpy as np
from machaon import rank1_constrained

W_new, delta = rank1_constrained(W, key, value, preserve=protected, max_rel_norm=0.02)
```

```python
from machaon.verify import check_protected

report = check_protected(W, delta, protected)
assert report["passed"] and report["max_drift"] < 1e-12
```

중요한 것은 두 번째 블록입니다. 첫 번째는 공식이고, 두 번째는 판정입니다.

## 실측 규모

| 모델 | 레이어 | 평균 rel.Δ | rewrite | rephrase | WikiText-2 |
|---|---|---|---|---|---|
| GPT-2 124M | 8 / 12 | 0.0135 | 0.500 | 0.259 | 1.066 |
| Qwen2.5-0.5B | 20 / 24 | 0.0133 | 0.375 | 0.275 | 1.030 |
| Qwen2.5-1.5B | 20 / 28 | **0.0065** | 0.500 | 0.273 | **1.005** |
| Qwen2.5-3B | 34 / 36 | 0.0072 | 0.625 | **0.327** | 1.009 |
| Qwen2.5-7B (int8) | 26 / 28 | **0.0043** | **1.000** | **0.929** | **1.007** |
| Llama-3.1-8B (int4) | 30 / 32 | 0.0138 | 0.500 | 0.500 | 1.031 |

## 문서

- [설치](getting-started/installation.md)
- [빠른 시작](getting-started/quickstart.md)
- [사실 편집하기](guides/edit-a-fact.md)
- [편집 평가하기](guides/evaluate.md)
- [벤치마크와 재현](reference/benchmarks.md)
- [Python API](reference/api.md)
- [MCP 서버](reference/mcp.md)
- [웹 콘솔](reference/web.md)

## 상태

Alpha이지만 코어는 실측되었습니다. 수학, 트랜잭션, 스토리지, 게이트, 런타임, safetensors 어댑터, OpenWSP v1을 구현했고 Python 3.9–3.13에서 테스트합니다. 실제 모델은 GPT-2 124M, Qwen2.5 0.5B / 1.5B / 3B / 7B, Llama-3.1-8B를 포함합니다. 7B/8B는 int8/int4 가중치와 국소 밀집 윈도우를 사용합니다. 7개의 CI job이 lint, 테스트 매트릭스, 어댑터와 커버리지, MCP 전송, 웹 콘솔, 문서 빌드, wheel/sdist 설치를 담당합니다. 다음 단계는 [로드맵](roadmap.md)을 참고하세요.
