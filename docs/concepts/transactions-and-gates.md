# Transactions and gates

## Transactions

An edit that cannot be undone is not an edit; it is a deployment.

```python
ws = Workspace()
ws.register("layer.8.mlp.c_proj", W)

txn = ws.begin("eiffel tower -> rome")
txn.apply("layer.8.mlp.c_proj", delta)
txn.commit(verdict)
```

- `apply` copies the original array before writing. Two edits to the same tensor
  in one transaction are undone in reverse order.
- `rollback()` restores every original byte exactly.
- `commit(verdict)` rolls back when `verdict.passed` is false. Passing `None`
  commits unconditionally — do that only when you have measured nothing and
  accept the risk.

A closed transaction refuses further operations; you cannot commit twice or
apply after a rollback.

## Gates

Gates are how an edit earns the right to be committed.

| gate | threshold | protects |
|---|---|---|
| `rewrite_acc` | ≥ 0.95 | the edit landed |
| `paraphrase_acc` | ≥ 0.60 | it generalises past the exact prompt |
| `unrelated_acc` | ≥ 1.00 | nothing else changed |
| `kl_p95` | ≤ 0.05 | distribution drift on unrelated prompts |
| `kl_max` | ≤ 0.50 | no single catastrophic outlier |
| `mmlu_delta` | ≥ −0.01 | capability preserved |
| `ppl_ratio` | ≤ 1.02 | no global damage |
| `longform_contradictions` | = 0 | long generation stays coherent |

## Fail closed

```python
from machaon.hygieia import evaluate

verdict = evaluate(metrics)
verdict.passed      # False when any metric is missing
verdict.missing     # ['mmlu_delta', ...]
```

A missing metric is a failure. Otherwise the easiest way to pass the suite is to
stop measuring, and every tool that makes measurement optional eventually gets
used without it.

For a narrow experiment you can pass an explicit gate subset — but say so:

```python
evaluate(metrics, {"rewrite_acc": (">=", 0.99), "ppl_ratio": ("<=", 1.05)})
```

## Reading a verdict

```
FAIL
  missing: longform_contradictions
  kl_p95 = 0.113 violates <= 0.05
```

Two different problems. `missing` means you did not measure; the other means you
measured and failed. Fix them in that order.
