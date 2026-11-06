# 체크포인트 저장하기

`machaon.adapters.Checkpoint`는 safetensors를 읽고 다시 쓰며 소스에는 절대 손대지 않습니다.

## 열기

```python
from machaon.adapters import Checkpoint

ckpt = Checkpoint("/models/qwen")            # index가 있는 디렉터리 또는 파일
ckpt.names()[:3]
ckpt.info("model.layers.3.mlp.down_proj.weight")
ckpt.find(layer=3, suffix="down_proj.weight")
```

지원 레이아웃:

- 단일 `.safetensors` 파일
- 정확히 하나를 포함한 디렉터리
- `model.safetensors.index.json`이 있는 샤드 체크포인트

## 로드 없이 검사

```python
ckpt.summary()
# {'tensors': 339, 'parameters': 1543714304, 'dtypes': {'BF16': ...}, 'shards': 1, ...}
```

헤더만 읽습니다. 수 GB 모델을 로드하기 전에 유용합니다.

## 모델 다운로드

```bash
python scripts/fetch_model.py --repo Qwen/Qwen2.5-1.5B --dest ./qwen15b
```

`huggingface.co`, `hf-mirror.com`, ModelScope 순으로 시도하고, 전송을 맹신하지 않고 각 `.safetensors` 헤더를 검증합니다.

## 읽기

```python
W = ckpt.read(name)                # float64, bf16/fp16에서 정확히 확장
W = ckpt.read(name, dtype=None)    # 저장 dtype 유지
```

읽기는 지연식: 한 번에 tensor 하나라서 7B 샤드 체크포인트를 메모리에 전부 올리지 않고 스트리밍합니다.

## tensor 하나, 샤드 하나 쓰기

두 벌을 들 수 없는 체크포인트에서는 tensor가 든 샤드만 다시 쓰고 나머지는 하드링크합니다:

```python
report = ckpt.write_tensor(name, W + delta, "/models/edited")
# {'shard': 'model-00004-of-00004.safetensors', 'linked': [...3 files...], 'copied': []}
```

피크 메모리는 샤드 하나. 7.6B 모델에서 14 GB가 아니라 약 4 GB, 쓰기는 2.3초.

## 쓰기

```python
report = ckpt.write_with({name: delta}, "/models/edited")
```

- 소스는 수정되지 않음;
- 건드리지 않은 tensor는 비트 단위 복사;
- 편집한 tensor는 float64로 더한 뒤 원래 dtype으로 변환;
- 샤드 체크포인트는 레이아웃과 index를 유지.

`report["tensors"][name]`에 상대 변화와 변경 항목 수가 들어갑니다.

## 나쁜 쓰기 거부

```python
from machaon.refusals import check_tensor

check_tensor(name, value, original,
             frozen={"model.embed_tokens.weight", "lm_head.weight"})
```

규칙: 동결 tensor, 모양 불일치, 비유한 값, bfloat16 오버플로, tensor 전체 스칼라 재스케일(모델을 조용히 줄이는 방법).
