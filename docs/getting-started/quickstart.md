# Quickstart

Five minutes. The first two steps run offline; the teaching step uses a local
checkpoint.

## 0 · One command from zero

```bash
machaon first
```

On a fresh machine this downloads SmolLM2-135M (~270 MB), opens it, teaches
"The Eiffel Tower is in the city of -> Rome", and saves the accepted edit to
`./models/SmolLM2-135M-machaon`. `--say` changes the fact, `--model` chooses an
existing checkpoint, and `--no-save` keeps the edit in memory.

## 1 · Teach a specific checkpoint

The shortest path from a checkpoint to a verified edit is one command:

Get the 0.5B checkpoint once (about 1 GB, with mirror fallback):

```bash
machaon fetch --starter qwen05b --dest ./models/Qwen2.5-0.5B
machaon fetch --list          # smol135m / qwen05b / gpt2
```

The downloader uses the standard library only, validates every safetensors
header, and resumes existing files.

```bash
machaon start
machaon intent "The Eiffel Tower is in the city of -> Rome"

machaon teach --model ./models/Qwen2.5-0.5B \
  --say "The Eiffel Tower is in the city of -> Rome"
```

`intent` compiles the sentence into the exact `prompt` and `answer` without
loading a model. `teach` then:

1. checks whether the model already returns the answer;
2. captures an unrelated baseline panel and projects the edit away from it;
3. tries a depth window of layers, applying and rolling back each candidate;
4. re-solves the best layer and applies it inside the budget;
5. measures the target token, top-1 preservation and KL; and
6. keeps the edit only when every required check passes. A failed check
   restores the original bytes.

The default output is a sibling directory named `<checkpoint>-machaon`, so the
source checkpoint stays untouched. Use `--dry-run` to search without writing,
`--no-save` to keep the edit in memory, and `--layers 8,10,12` when you already
know the site.

Supported sentence forms include:

```text
The Eiffel Tower is in the city of -> Rome
prompt: The Eiffel Tower is in the city of
answer:  Rome
when asked "The Eiffel Tower is in the city of", answer " Rome"
教会模型：北京的首都是 → 北京
当问到「北京的首都是」时，回答「北京」
```

The arrow and labelled forms are exact. Plain sentences are also accepted and
are marked medium confidence because the split point is an assumption. The
local checks speak about the target and the declared panel; a benchmark-level
claim still needs the full gate report.

Teaching pins the **first token** of the answer. A multi-token answer is
reported as a warning; the rest of the continuation is a benchmark question,
not a guarantee of this local check.
### Teach a small list

```bash
cat > facts.txt <<'EOF'
The Eiffel Tower is in the city of -> Rome
The capital of France is -> Paris
EOF

machaon teach-many --model ./models/Qwen2.5-0.5B --facts facts.txt --out ./models/qwen-many
```

Each fact gets its own layer search and local checks. The chain keeps the last
globally safe checkpoint at `<out>/final`, records every step in
`<out>/report.json`, and stops without touching earlier facts when one is
rejected. `--continue-on-reject` skips a bad fact; `--keep-steps` keeps the
whole chain.


Use the manual steps below when you want to choose the matrix, the key and the
protected set yourself.

## 2 · Run the demo

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

Read the third column. The naive edit moves the protected keys by 8.7e-2; the
constrained edit leaves them at 3.1e-16. The last digit can move at the 1e-16
level across BLAS implementations.

To exercise the complete program path offline — generate, run, save, reload and
verify a report — run:

```bash
python examples/tiny_program.py
```

It builds a tiny random GPT-2 locally and downloads nothing. This is a
plumbing check, not a language-model result.

## 3 · Edit a matrix you own

```python
import numpy as np
from machaon import rank1_constrained
from machaon.verify import check_protected

rng = np.random.default_rng(0)
W = rng.normal(0, 0.5, size=(64, 48))      # (d_out, d_in)
keys = rng.normal(size=(16, 48))           # must not move
k = rng.normal(size=48)                    # must memorise
v = W @ k + rng.normal(0, 1.0, size=64)    # must produce

W_new, delta = rank1_constrained(W, k, v, preserve=keys, max_rel_norm=0.05)

print(check_protected(W, delta, keys))
```

```
{'passed': True, 'max_drift': 3.1e-16, ...}
```

## 4 · Put it in a transaction

```python
from machaon import Workspace

ws = Workspace()
ws.register("layer.8.mlp.c_proj", W)

txn = ws.begin("first edit")
txn.apply("layer.8.mlp.c_proj", delta)

# anything with a boolean `.passed` works as a verdict
txn.commit()          # or txn.rollback()
```

`commit(verdict)` refuses when `verdict.passed` is false and restores the
original bytes.

## 5 · Run it on a real model

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

## Where next

- [Edit a fact](../guides/edit-a-fact.md) — the full procedure, step by step.
- [Evaluate an edit](../guides/evaluate.md) — gates, KL, benchmark panels.
- [The maths](../MATH.md) — derivations, diagnostics and the operating envelope.
