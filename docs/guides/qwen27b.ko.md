# Qwen3.6-27B 가중치 공간 프로그래밍

Qwen3.6-27B는 비전 인코더가 있는 멀티모달 체크포인트입니다. 텍스트
디코더는 64개 레이어이며 MLP 출력 투영은 다음 위치에 있습니다.

```text
model.language_model.layers.<layer>.mlp.down_proj.weight
```

Machaon은 이 멀티모달 경로를 먼저 찾고, 없으면 일반 `model.layers`로
폴백합니다.

## 1 · 호환 runtime 설치

Qwen3.6에는 최신 Transformers와 `qwen3_5` 지원이 필요합니다. 별도 환경을 권장합니다.

```bash
uv venv .venv-qwen36
source .venv-qwen36/bin/activate
uv pip install -e ".[hf]"
uv pip install -U "transformers[serving]" torchvision pillow optimum-quanto
```

`machaon.runtime.load_pretrained`는 config에 따라
`AutoModelForImageTextToText` 또는 `AutoModelForCausalLM`을 선택합니다.

## 2 · 다운로드와 검사

```bash
python scripts/fetch_model.py \
  --repo Qwen/Qwen3.6-27B \
  --dest ./qwen36-27b \
  --endpoint modelscope
```

전체 가중치를 로드하지 않고 header만 확인합니다.

```python
from machaon.adapters import Checkpoint

ckpt = Checkpoint("./qwen36-27b")
print(ckpt.summary())
print(ckpt.find(layer=60, suffix="down_proj.weight"))
```

## 3 · 메모리 작업점

| 모드 | 가중치 | 용도 |
|---|---:|---|
| bf16 | 약 54 GB | 대용량 메모리/GPU |
| int4 | 약 14 GB | 34 GB 머신 평가 |
| checkpoint-only | shard 1개(약 4 GB) | 이미 계산한 delta 저장 |

34 GB 환경에서는 int4와 로컬 dense window를 사용하고 56–60층 근처를 편집합니다.

```python
import torch
from transformers import AutoTokenizer, QuantoConfig
from machaon.runtime import HFModel, load_pretrained

tokenizer = AutoTokenizer.from_pretrained("./qwen36-27b")
model = load_pretrained(
    "./qwen36-27b",
    quantization_config=QuantoConfig(weights="int4"),
    device_map="cpu",
)
runtime = HFModel(model, tokenizer)
assert runtime.layer_path == "model.language_model.layers"
runtime.densify_from(60)
```

## 4 · 제약된 편집 풀기

```python
from machaon.procedures import solve_edit

neutral = [
    "The capital of Germany is",
    "Water freezes at a temperature of",
    "Two plus two equals",
    "The Moon orbits the",
]
proposal = solve_edit(
    runtime,
    "The Eiffel Tower is located in the city of",
    " Rome",
    layer=60,
    preserve_prompts=neutral,
    solver="gradient",
    steps=40,
    lr=0.02,
    max_rel_norm=0.005,
)
print(proposal.module, proposal.relative_delta)
```

첫 확인에는 `solver="direction"`도 사용할 수 있습니다.

## 5 · 검증과 shard 쓰기

```python
from machaon.adapters import Checkpoint
from machaon.verify import check_protected

W = runtime.read(proposal.module)
W_new = W + proposal.delta
report = check_protected(W, proposal.delta, protected_keys)
assert report["passed"]

ckpt = Checkpoint("./qwen36-27b")
ckpt.write_tensor(proposal.module, W_new, "./qwen36-edited")
```

대상 tensor가 있는 shard만 다시 쓰고 나머지는 하드링크합니다. 원본
체크포인트는 바뀌지 않습니다. 27B에서는 이 저장 경로가 실용적입니다.

## 6 · Session으로 관리

```python
from machaon.session import Session

session = Session(runtime, "./qwen36-27b")
session.baseline(neutral, held_out_documents)
session.propose(
    "The Eiffel Tower is located in the city of", " Rome", layer=60,
    preserve_prompts=neutral, solver="gradient", steps=40, lr=0.02,
    max_rel_norm=0.005,
)
session.apply()
metrics = session.evaluate(
    ["The Eiffel Tower is located in the city of"],
    [tokenizer(" Rome", add_special_tokens=False)["input_ids"][0]],
)
session.rollback()
```

`EditSuite`/`judge`에는 같은 모델의 before/after 상태나 checkpoint 스냅샷을
사용하고 27B를 두 번 로드하지 마십시오.

## 7 · 브라우저에서 같은 흐름 실행

로컬 콘솔은 같은 session과 checkpoint 허용 목록을 사용합니다.

```bash
pip install -e ".[web]"
machaon-web --allowed-root ./qwen36-27b --port 8765
```

출력된 URL을 열고 checkpoint header를 확인한 뒤 `dtype=auto`,
`quantize=int4`, `device=cpu`, 60층, 예산 `0.01`로 session을 엽니다.
중립 prompt를 baseline으로 캡처하고 proposal의 상대 변화량을 확인한 뒤
**Apply**를 누릅니다. 증거 열은 작업 상태를 스트리밍하고 평가 후 fail-closed
gate를 적용합니다. rollback은 원래 tensor 바이트를 복원합니다.
