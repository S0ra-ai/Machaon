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
