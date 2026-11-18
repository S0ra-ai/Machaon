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

## `first`

从零开始的最短路径。`--model` 不存在时，它用与 `machaon fetch` 相同的端点
回退和 header 校验下载所选 starter（默认 SmolLM2-135M）；然后打开检查点、
教一句事实，并保存通过的编辑。

```bash
machaon first
machaon first --say "法国的首都是 -> 柏林"
machaon first --model ./models/Qwen2.5-0.5B --out ./models/qwen-first
```

默认事实是 "The Eiffel Tower is in the city of -> Rome"，默认输出是
`<model>-machaon`。`--no-save` 只保留在内存中；`--json` 输出完整结果。
事实已应用或本来就正确时退出码 0，被本地检查拒绝时 1，输入或模型错误时 2。

## `fetch`

下载 safetensors 检查点，不导入 torch，也不加载任何权重 tensor。它依次通过
Hugging Face、`hf-mirror`、ModelScope 列出仓库，并在接受每个
`.safetensors` 前校验 header。已存在的文件会复用，因此中断后可以续传。

```bash
machaon fetch --list
machaon fetch --starter qwen05b --dest ./models/Qwen2.5-0.5B
machaon fetch --starter smol135m --dest ./models/SmolLM2-135M
machaon fetch --repo Qwen/Qwen2.5-0.5B --dest ./models/qwen
```

`--endpoint huggingface|hf-mirror|modelscope` 强制使用一个端点。`--json`
输出文件列表、分片数、tensor 数与参数数。没有 `.safetensors` 权重的仓库会被
直接拒绝，不会留下半下载目录。

## `start`

打印第一次编辑的最短路径，并报告已安装的可选运行时：

```bash
python -m machaon.cli start
python -m machaon.cli start --demo
```

## `intent`

把一句自然语言教学请求编译成精确的 `prompt` / `answer` 合同，不加载模型。
退出码 2 表示无法唯一确定一对 prompt/answer；错误信息会列出支持的句式。

```bash
python -m machaon.cli intent "埃菲尔铁塔位于 -> 罗马"
python -m machaon.cli intent "教会模型：北京的首都是 → 北京" --json
```

箭头和标签形式是精确的；普通句子也会被接受，但会标记为 `medium` 置信度，
因为切分点是假设。命令会同时打印等价的 `machaon teach` 命令。

## `teach`

运行完整的教学配方：基线、深度窗口逐层搜索、应用、目标/面板/KL 检查，
失败时自动回滚。

```bash
python -m machaon.cli teach --model ./models/Qwen2.5-0.5B \
  --say "埃菲尔铁塔位于 -> 罗马"
```

默认输出是同级目录 `<checkpoint>-machaon`，源检查点不会被修改。常用参数：

| 参数 | 含义 |
|---|---|
| `--say TEXT` | 用 `machaon intent` 编译这句话 |
| `--prompt P --answer A` | 直接给出合同 |
| `--layers 8,10,12` / `--layers 8-12` | 指定候选层 |
| `--max-layers 7` | 默认深度窗口的候选数量 |
| `--solver direction\|gradient` | 每个候选使用的 value solver |
| `--max-rel-norm 0.02` | 单次编辑的相对范数预算 |
| `--budget 0.10` | 会话累计预算 |
| `--max-kl 0.05` | KL p95 验收阈值 |
| `--min-panel-agreement 0.875` | 无关 prompt 保持 top-1 的最低比例 |
| `--preserve-file FILE` | 每行一个基线/投影 prompt |
| `--corpus-file FILE` | 每行一段文本，用于可选的困惑度检查 |
| `--out DIR` | 编辑后的检查点目录（已存在则拒绝） |
| `--no-save` | 只保留在内存中 |
| `--dry-run` | 只搜索和报告，不写入 |
| `--evidence-out DIR` | 写入带未编辑 headroom 的 WSP 证据包 |
| `--edit-out DIR` | 把已应用的 delta 导出为 OpenWSP v1 |
| `--report-out FILE` | 写入完整配方报告 JSON |
| `--json` | 机器可读输出 |

退出码：`0` 已应用 / 无需修改 / 试运行，`1` 被拒绝并已回滚，`2` 输入错误或模型错误。

`--out`、`--evidence-out`、`--edit-out` 的目标已存在时会拒绝运行，避免覆盖上一次的产物。

## `teach-many`

把一组事实教进一条 checkpoint 链。每条事实都在一个从上一个已接受检查点加载的
干净 session 上运行完整的单条配方；源检查点不会被修改。

```bash
machaon teach-many --model ./models/Qwen2.5-0.5B \
  --facts facts.txt --out ./models/qwen-many
```

`facts.txt` 每行一条教学句子（语法与 `machaon intent` 相同），也可以是
JSON / JSONL 的 `{"say": ...}` 或 `{"prompt": ..., "answer": ...}` 列表。
命令把最后一条全局安全的检查点放在 `<out>/final`，把每一步证据写入
`<out>/report.json`。某条事实被拒绝或累计漂移检查失败时退出码为 1。
`--continue-on-reject` 会记录失败并继续；`--keep-steps` 保留所有步骤而不是
只保留最后一条安全步骤，此时 final 是最后一个 `step-NNN` 目录而不是
`<out>/final`。累计 `--budget` 作用于整条链，而不是每条事实。

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

## `wsp`

对 JSON 报告应用 WSP 验收协议。只有候选 gate 通过且标签为 `repair` 或
`acquisition` 时退出码为 0；其他标签均退出 1。

```bash
python -m machaon.cli wsp examples/wsp_report.json
```

报告包含 `base`、`candidate`，可选 `controls`、`headroom`、`mechanism`、
`gates` 与 `protocol`。完整分类、control gap、headroom gap 与机制门以 JSON 输出。

`--save DIR` 写出内容寻址的 evidence bundle；`--metadata FILE` 附加一个
JSON 对象。

```bash
python -m machaon.cli wsp examples/wsp_report.json --save evidence/
python -m machaon.cli wsp-verify evidence/
```

### WSP benchmark

```bash
python -m machaon.cli wsp-bench examples/wsp_benchmark.json
python -m machaon.cli wsp-bench measured.json --require-measured
```

每个 case 声明 `failure_type`（`obstruction` 或 `capability_gap`）和
`expected_repairable`。命令输出逐例 WSP 证据与一致性/混淆矩阵摘要；任一样本
与标签不一致时退出码为 1。仓库自带 fixture 是合成的，不能当作真实模型测量结果。
`--require-measured` 会在存在非 measured case 时以退出码 2 拒绝。
`examples/wsp_measured_smoke.json` 是由
`scripts/make_wsp_reference_case.py` 生成的 measured 阈值分类器参考用例；
它用于验证 provenance gate，不是 LLM 能力结论。

### 规划

```bash
python -m machaon.cli plan examples/plan_obstruction.json
```

输入声明 `failure_type`，以及 `layer_scores` 或因果 patching 的 `patch_report`。
输出是 `edit`、`diagnose` 或 `do_not_edit` 路线。没有超过阈值的位点时，不会猜一个
layer 出来。

### 诊断

```bash
python -m machaon.cli diagnose examples/diagnose_obstruction.json
```

输入提供 `base`、可选 `headroom`、可选 `mechanism`、可选诊断 `protocol` 与可选
`plan` 参数。输出保守的失败类型；若提供 plan 参数，同时给出对应的
edit / diagnose / do-not-edit 计划。证据缺失或矛盾时保持 `unknown`。

`wsp-verify` 校验格式/版本并重算 evidence SHA-256。

## `binding`

校验 OpenWSP edit 与 WSP evidence bundle 是否仍然互相指向：

```bash
python -m machaon.cli binding out/edit evidence/
```

退出码 1 表示 edit 地址、payload 地址或 evidence 绑定失败。

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
