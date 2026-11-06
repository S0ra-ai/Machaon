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
