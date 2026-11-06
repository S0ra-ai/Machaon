# 命令行

```bash
python -m machaon <command>
```

退出码是契约的一部分，可以直接作为 CI 步骤：

| 退出码 | 含义 |
|---|---|
| 0 | 成功 / gate 通过 |
| 1 | gate 拒绝该编辑，或校验失败 |
| 2 | 输入格式错误、文件缺失、检查点不支持 |

## `demo`

运行合成 rank-one demo。无模型、无网络。

```bash
python -m machaon.cli demo
```

## `validate`

校验 edit spec。spec 无效时退出码 1，JSON 损坏时退出码 2。

```bash
python -m machaon.cli validate examples/edit_spec.json
# OK  rank1_constrained -> model.layers.14.mlp.down_proj (budget 0.020)
```

## `gate`

对 metrics JSON 应用验收门。退出码 1 表示不应提交——因此可以直接放进 CI。

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

列出 safetensors 检查点中的 tensor。

```bash
python -m machaon.cli inspect /models/qwen --suffix down_proj.weight
python -m machaon.cli inspect /models/qwen --layer 3 --limit 5
python -m machaon.cli inspect /models/qwen --layer 3 --json
```

## `doctor`

报告已安装的能力和版本，适合贴进 issue。

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

运行测试、demo、spec 校验、两个 gate 用例和 lint。
