# Quickstart

Five minutes, no download. Every command here runs offline.

## 1 · Run the demo

```bash
python -m machaon.cli demo
```

```
op                         rel.delta   target err   preserve drift probe drift p95
----------------------------------------------------------------------------------
rank1 (naive)                 0.0509     1.60e-16         8.71e-02          0.0902
constrained (no cap)          0.0730     2.02e-16         3.82e-16          0.1394
constrained (capped)          0.0200     2.84e-01         2.52e-16          0.0382
```

Read the third column. The naive edit moves the protected keys by 8.7e-2; the
constrained edit leaves them at 3.8e-16.

## 2 · Edit a matrix you own

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

## 3 · Put it in a transaction

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

## 4 · Run it on a real model

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
PASS (6 gates)
```

## Where next

- [Edit a fact](../guides/edit-a-fact.md) — the full procedure, step by step.
- [Evaluate an edit](../guides/evaluate.md) — gates, KL, benchmark panels.
- [The maths](../MATH.md) — derivations and failure modes.
