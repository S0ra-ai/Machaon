# 설치

## 요구 사항

| | |
|---|---|
| Python | 3.9, 3.10, 3.11, 3.12, 3.13 |
| 코어 | NumPy만 |
| 실제 모델 | PyTorch + transformers + safetensors |
| 벤치마크 | pyarrow |
| MCP 서버 | `mcp` SDK, Python 3.10+ |
| 브라우저 콘솔 | FastAPI + uvicorn |

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
uv pip install -e ".[web]"      # 로컬 브라우저 콘솔
uv pip install -e ".[dev,docs]" # 테스트, lint, 커버리지, 문서
```

## 검증

```bash
./scripts/ci.sh
```

테스트, 합성 데모, spec 검증, 두 개의 게이트 케이스, lint를 실행합니다. 네트워크와 GPU가 필요 없습니다.

예상되는 마지막 출력:

```
Ran 583 tests ...
ALL CHECKS PASSED
```

## 모델 다운로드

`machaon fetch`는 가중치를 읽지 않고 safetensors 체크포인트를 받습니다.
Hugging Face, `hf-mirror`, ModelScope 순으로 시도하고 각 safetensors 헤더를
검증하며 기존 파일을 재사용합니다.

```bash
machaon fetch --list
machaon fetch --starter qwen05b --dest ./models/Qwen2.5-0.5B
machaon fetch --repo Qwen/Qwen2.5-1.5B --dest ./qwen15b
```

`HF_ENDPOINT`는 Transformers 로더에는 계속 적용되지만, 내장 다운로더는 자체
폴백을 사용합니다. `.safetensors` 가중치가 없는 저장소는 반쯤 받은 디렉터리를
남기지 않고 거부합니다.

벤치마크 데이터는 따로 받습니다:

```bash
python scripts/fetch_benchmarks.py
```

파일은 `$MACHAON_BENCH`(기본 `~/.cache/machaon/bench`)에 저장되며, 각 파일 옆 `.provenance.json`에 소스, revision, 라이선스, 항목 수, SHA-256을 기록합니다.

## 문제 해결

**macOS에서 `RuntimeWarning: divide by zero encountered in matmul`** —— Accelerate의 부동소수점 플래그 특성입니다. Machaon 자체 커널에서는 억제합니다. 사용자 코드에서 보인다면 플랫폼 동작이며 수학 오류가 아닙니다.

**`this runtime has no tokenizer`** —— ID 기반 runtime에서 텍스트 메서드를 호출했습니다. token id를 넘기거나 `HFModel(model, tokenizer)`를 구성하세요.
