# 설치

## 요구 사항

| | |
|---|---|
| Python | 3.9, 3.10, 3.11, 3.12, 3.13 |
| 코어 | NumPy만 |
| 실제 모델 | PyTorch + transformers + safetensors |
| 벤치마크 | pyarrow |
| MCP 서버 | `mcp` SDK, Python 3.10+ |

## 설치

```bash
git clone https://github.com/S0ra-ai/Machaon.git
cd Machaon
uv venv && uv pip install -e ".[dev]"
```

선택 의존성:

```bash
uv pip install -e ".[hf]"       # torch + transformers + safetensors
uv pip install -e ".[bench]"    # pyarrow, MMLU / WikiText 로더
uv pip install -e ".[mcp]"      # MCP SDK, Python 3.10+
uv pip install -e ".[dev,docs]" # 테스트, lint, 커버리지, 문서
```

## 검증

```bash
./scripts/ci.sh
```

테스트, 합성 데모, spec 검증, 두 개의 게이트 케이스, lint를 실행합니다. 네트워크와 GPU가 필요 없습니다.

예상되는 마지막 출력:

```
Ran 202 tests ...
ALL CHECKS PASSED
```

## 모델 다운로드

`huggingface.co`에 접근할 수 없다면 클라이언트를 미러로 돌립니다:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

접근 가능한 엔드포인트로 모델을 받습니다:

```bash
python scripts/fetch_model.py --repo Qwen/Qwen2.5-1.5B --dest ./qwen15b
```

Hugging Face, 미러, ModelScope 순으로 시도하고 전송 후 safetensors 헤더를 검증합니다.

벤치마크 데이터는 따로 받습니다:

```bash
python scripts/fetch_benchmarks.py
```

파일은 `$MACHAON_BENCH`(기본 `~/.cache/machaon/bench`)에 저장되며, 각 파일 옆 `.provenance.json`에 소스, revision, 라이선스, 항목 수, SHA-256을 기록합니다.

## 문제 해결

**macOS에서 `RuntimeWarning: divide by zero encountered in matmul`** —— Accelerate의 부동소수점 플래그 특성입니다. Machaon 자체 커널에서는 억제합니다. 사용자 코드에서 보인다면 플랫폼 동작이며 수학 오류가 아닙니다.

**`this runtime has no tokenizer`** —— ID 기반 runtime에서 텍스트 메서드를 호출했습니다. token id를 넘기거나 `HFModel(model, tokenizer)`를 구성하세요.
