# CLI

```bash
python -m machaon.cli <command>
```

## `demo`

Runs the synthetic rank-one demo. No model, no network.

```bash
python -m machaon.cli demo
```

## `validate`

Validates an edit spec. Exit code 1 on an invalid spec.

```bash
python -m machaon.cli validate examples/edit_spec.json
# OK  rank1_constrained -> model.layers.14.mlp.down_proj (budget 0.020)
```

## `gate`

Applies the acceptance gates to a metrics JSON file. Exit code 1 when the edit
should not be committed — which makes it usable as a CI step.

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

Lists tensors in a safetensors checkpoint.

```bash
python -m machaon.cli inspect /models/qwen --suffix down_proj.weight
python -m machaon.cli inspect /models/qwen --layer 3 --limit 5
```

## `--version`

```bash
python -m machaon.cli --version
```

## The three-command CI

```bash
./scripts/ci.sh
```

Runs the tests, the demo, spec validation, both gate cases, and lint.
