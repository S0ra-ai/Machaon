# Machaon

**Direct weight surgery for LLMs — edit the weights, not the gradients.**

Machaon changes a model's parameters directly, then proves the change did not
break anything: what the edit was allowed to touch, what it actually moved, and
how to put the model back if the answer is wrong.

```
              edit spec (DSL)
                    │
        ┌───────────▼───────────┐
        │ linalg                │  rank-1 · null-space · ablation
        └───────────┬───────────┘
                    │ delta
        ┌───────────▼───────────┐
        │ transaction · store   │  rollback · content addressing
        └───────────┬───────────┘
     ┌──────────────┼──────────────┐
┌────▼────┐   ┌─────▼─────┐   ┌────▼─────┐
│ runtime │   │ hygieia   │   │ interop  │
│ HF      │   │ gates     │   │ OpenWSP  │
└─────────┘   └───────────┘   └──────────┘
```

## Start here

<div class="grid cards" markdown>

-   **New to weight editing?**

    ---

    Read [Weight editing](concepts/weight-editing.md) for what an edit is and
    why locality is the hard part, then follow the
    [quickstart](getting-started/quickstart.md).

-   **Want numbers?**

    ---

    [Benchmarks](reference/benchmarks.md) lists every dataset, its licence, and
    the exact protocol. [Results](RESULTS.md) has the current numbers.

-   **Integrating with another tool?**

    ---

    [OpenWSP interop](guides/openwsp-interop.md) — the edit format is
    byte-identical, so edits move both ways.

-   **Reading the API?**

    ---

    [Python API](reference/api.md) documents every public function and the
    invariants it keeps.

</div>

## The three guarantees

| guarantee | how | checked by |
|---|---|---|
| protected keys do not move | `ΔW k⊥ = 0` by construction | `verify.check_protected` |
| a rejected edit is undone exactly | originals copied before applying | `transaction.rollback` |
| an unmeasured edit is rejected | missing metrics count as failures | `hygieia.evaluate` |

## A 30-second taste

```python
import numpy as np
from machaon import rank1_constrained

W_new, delta = rank1_constrained(W, key, value, preserve=protected, max_rel_norm=0.02)
```

```python
from machaon.verify import check_protected

report = check_protected(W, delta, protected)
assert report["passed"] and report["max_drift"] < 1e-12
```

That second block is the point. The first block is a formula; the second is a
verdict.

## Status

Alpha. The maths, transactions, storage, gates, runtime, safetensors adapter and
OpenWSP format are implemented and tested (161 tests). Real-model evidence
currently stops at GPT-2 124M. See the [roadmap](roadmap.md).
