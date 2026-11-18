# 快速开始

五分钟。前两步无需下载；教学步骤使用本地检查点。

## 0 · 一句话教学

从检查点到一次经过验证的编辑，最短路径只有一条命令：

先下载一次 0.5B 检查点（约 1 GB，自动尝试镜像）：

```bash
machaon fetch --starter qwen05b --dest ./models/Qwen2.5-0.5B
machaon fetch --list          # smol135m / qwen05b / gpt2
```

下载器只依赖标准库，会校验每个 safetensors header，并复用已下载的文件。

```bash
machaon start
machaon intent "埃菲尔铁塔位于 -> 罗马"

machaon teach --model ./models/Qwen2.5-0.5B \
  --say "埃菲尔铁塔位于 -> 罗马"
```

`intent` 只把句子编译成精确的 `prompt` 与 `answer`，不加载模型。`teach` 随后：

1. 检查模型是否已经会回答；
2. 自动捕获一组无关基线 prompt，并把编辑投影到它们的零空间；
3. 在一段深度窗口内逐层尝试，每次应用后评估并回滚；
4. 重新求解最佳层并在预算内应用；
5. 测量目标 token、无关 prompt 的 top-1 保持率和 KL；
6. 只有所有必选检查通过才保留编辑；任何检查失败都会还原原始字节。

默认输出是同级目录 `<checkpoint>-machaon`，源检查点不会被修改。
`--dry-run` 只搜索不写入，`--no-save` 只保留在内存中，已知层时可用
`--layers 8,10,12`。

支持的句式包括：

```text
埃菲尔铁塔位于 -> 罗马
提示词：埃菲尔铁塔位于
回答:  罗马
当问到「北京的首都是」时，回答「北京」
教会模型：北京的首都是 → 北京
```

箭头和标签形式是精确的；普通句子也会被接受，但会标记为中等置信度，因为
切分点是假设。本地检查只针对目标与声明的基线面板；基准级结论仍需要完整
验收报告。

教学固定的是答案的**第一个 token**。多 token 答案会给出警告；后续内容属于
基准问题，不是这次本地检查的保证。
### 教一组事实

```bash
cat > facts.txt <<'EOF'
埃菲尔铁塔位于 -> 罗马
法国的首都是 -> 巴黎
EOF

machaon teach-many --model ./models/Qwen2.5-0.5B --facts facts.txt --out ./models/qwen-many
```

每条事实都有自己的选层搜索和本地检查。整条链把最后一条全局安全的检查点放在
`<out>/final`，把每一步写入 `<out>/report.json`；某条被拒绝时停止，并保留
之前的事实。`--continue-on-reject` 跳过坏事实；`--keep-steps` 保留整条链。


需要自己选择矩阵、key 和保护集时，再使用下面的手动步骤。

## 1 · 运行 demo

```bash
python -m machaon.cli demo
```

```
op                         rel.delta   target err   preserve drift probe drift p95
----------------------------------------------------------------------------------
rank1 (naive)                 0.0509     1.60e-16         8.71e-02          0.0902
constrained (no cap)          0.0730     2.02e-16         3.11e-16          0.1394
constrained (capped)          0.0200     2.84e-01         8.55e-17          0.0382
```

看第三列。朴素编辑把受保护 key 移动了 8.7e-2；受约束编辑把它们留在 3.8e-16。

## 2 · 编辑一个你拥有的矩阵

```python
import numpy as np
from machaon import rank1_constrained
from machaon.verify import check_protected

rng = np.random.default_rng(0)
W = rng.normal(0, 0.5, size=(64, 48))      # (d_out, d_in)
keys = rng.normal(size=(16, 48))           # 不能移动
k = rng.normal(size=48)                    # 必须记住
v = W @ k + rng.normal(0, 1.0, size=64)    # 必须输出

W_new, delta = rank1_constrained(W, k, v, preserve=keys, max_rel_norm=0.05)

print(check_protected(W, delta, keys))
```

```
{'passed': True, 'max_drift': 3.1e-16, ...}
```

## 3 · 放进事务

```python
from machaon import Workspace

ws = Workspace()
ws.register("layer.8.mlp.c_proj", W)

txn = ws.begin("first edit")
txn.apply("layer.8.mlp.c_proj", delta)

# 任何带布尔 `.passed` 的对象都可以作为 verdict
txn.commit()          # 或 txn.rollback()
```

`commit(verdict)` 在 `verdict.passed` 为 false 时拒绝提交并还原原始字节。

## 4 · 在真实模型上运行

```bash
export HF_ENDPOINT=https://hf-mirror.com
python examples/gpt2_rome.py
```

```
model      : gpt2 (12 layers)
edit       : transformer.h.8.mlp.c_proj.weight
delta      : rel 0.0153
  rewrite_acc  1
  paraphrase_acc 1
  kl_p95       0.000614724
  ppl_ratio    1.00079
PASS (5 gates)
```

## 下一步

- [编辑一个事实](../guides/edit-a-fact.md) —— 完整流程，逐步拆解。
- [评估一次编辑](../guides/evaluate.md) —— 验收门、KL、基准面板。
- [数学原理](../MATH.md) —— 推导、诊断与适用范围。
