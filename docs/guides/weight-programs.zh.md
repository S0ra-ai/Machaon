# 权重程序

权重程序是一份小小的文本文件，用来描述一次完整、可审查的权重空间操作：

- 从哪个检查点开始；
- 模型要学会哪些事实；
- 哪些提示必须保持不变；
- 模型最多允许改变多少；
- 保存后是否重新加载并复验。

它是 `machaon first`、`machaon teach-many` 和浏览器控制台同一条闭环的可分享形态。
权重程序不会引入第二套编辑引擎：它只解析路径与策略，调用教学链，然后检查真正写出的字节。

## 一个完整的程序

保存为 `hello.machaon`：

```text
version: 1
name: Hello, weight space

model: ./models/SmolLM2-135M
fetch: smol135m
output: ./models/hello-weight-space

budget: 0.10
max_rel_norm: 0.02
verify: true

teach: The Eiffel Tower is in the city of -> Rome
teach: The capital of France is -> Paris

protect: The capital of Germany is
protect: Water freezes at a temperature of
```

不下载、不加载模型，先检查解析后的合同：

```bash
machaon program hello.machaon --plan
```

运行：

```bash
machaon program hello.machaon
```

只有 `model` 不存在且 `fetch` 指定了 starter 时才会下载。每条事实独立搜索并做局部验收；
链路把最后一个全局安全的检查点放在 `<output>/final`，同时写入 `report.json` 和
`<output>/program-report.json`。

`verify: true` 会在保存后重新加载该检查点，再检查每一条事实。保存成功但复验失败时，
程序状态是 `verification_failed`，退出码非零。只有你打算另行复验字节时才使用 `--no-verify`。

## 字段

只有 `model`、`output` 和至少一条 `teach` 是必填的。

| 字段 | 默认值 | 含义 |
|---|---|---|
| `model` | 必填 | 检查点目录；相对路径相对于程序文件解析 |
| `output` | 必填 | 新检查点容器；已存在且非空的目录会被拒绝 |
| `fetch` | 无 | `model` 缺失时下载的 starter：`smol135m`、`qwen05b` 或 `gpt2` |
| `teach` | 必填，可重复 | 一句教学请求，或 JSON 的 `prompt` / `answer` 对象 |
| `protect` | 无，可重复 | 用于基线、投影和保护面板的无关提示 |
| `budget` | `0.10` | 所有事实累计的相对范数预算 |
| `max_rel_norm` | `0.02` | 单条事实的相对范数上限 |
| `max_kl` | `0.05` | 受保护面板的 KL p95 上限 |
| `min_panel_agreement` | `0.875` | 必须保持 top-1 的受保护提示比例 |
| `layers` | 自动深度窗口 | `18,20,22` 或 `18-22` |
| `solver` | `direction` | `direction` 或 `gradient` |
| `dtype` | `float32` | `auto`、`float32`、`float16` 或 `bfloat16` |
| `quantize` | 无 | `int4` 或 `int8`；要求 `dtype: auto` |
| `device` | 运行时默认 | `cpu`、`mps`、`cuda` 或运行时支持的值 |
| `verify` | `true` | 重新加载写出的检查点并检查每条事实 |
| `continue_on_reject` | `false` | 记录被拒事实并继续 |
| `keep_steps` | `false` | 保留每一步检查点，而不只保留安全 final |

`#` 开始注释；空行忽略。字段写成 `key: value` 或 `key = value`；
`teach` 和 `protect` 可以出现多次。

## 相对路径

`model` 和 `output` 的相对路径相对于程序文件所在目录解析。这样复制程序文件和它的
`models/` 目录到另一台机器时，合同不会悄然改变。

`--out DIR` 只覆盖本次运行的输出。程序哈希和程序声明的原输出仍会留在报告里，
因此覆盖不会偷偷改写共享合同：

```bash
machaon program hello.machaon --out ./models/experiment-2
```

## JSON 形式

同一份合同也可以由程序生成成 JSON。事实可以是句子、`{"say": "..."}`，
或显式的 `{"prompt": "...", "answer": "..."}`：

```json
{
  "version": 1,
  "model": "./models/SmolLM2-135M",
  "fetch": "smol135m",
  "output": "./models/hello-weight-space",
  "facts": [
    "The Eiffel Tower is in the city of -> Rome",
    {"prompt": "The capital of France is", "answer": " Paris"}
  ],
  "protect": ["Two plus two equals"],
  "verify": true
}
```

两种形式产出相同的 `program-report.json`：包含程序 SHA-256、解析后的策略、
每一步教学结果和复验结果。分享或审查程序时，这份文件就是审计记录。

## 它不声称什么

本地检查证明声明的事实成立、受保护面板没有越界；它们不能替代基准测试。
程序报告是这次编辑的证据，不是对模型的普遍结论。
