# Weight editing

An **edit** is an additive change to one or more weight matrices:

```
W' = W + ΔW
```

Machaon restricts `ΔW` to a *derivable* family, so that the effect of an edit can
be reasoned about rather than hoped for.

## The action space

| level | what the agent chooses | what the maths does | status |
|---|---|---|---|
| L0 | which procedure, which layer | nothing yet | ✅ `procedures.rome` |
| L1 | a Python function over tensors | executes it | next |
| L2 | a target key/value pair | rank-one solve | ✅ `rank1_constrained` |
| L3 | an interpretable direction | ablation / injection | ✅ `ablate_*`, `bake_direction` |
| L4 | a sparse set of scalars | literal assignment | via `Workspace` |
| L5 | structure: prune, merge, rewire | handled by merge/prune tools; their deltas audit cleanly here | interop |

Most published work lives at L2 and L3. L0 is what turns an algorithm into a
tool: the model decides *where* and *how much*, the maths decides *what*.

## What makes an edit good

Three properties, in order of how often they are violated:

### 1. Locality

Changing the answer to one prompt must not change the answer to unrelated
prompts. This is the property that fails first and loudest. Machaon addresses it
with a null-space projection over a protected key set — see
[null-space projection](null-space.md).

Measured as `‖ΔW kᵢ‖ / ‖W kᵢ‖` for protected keys, and as KL divergence on
unrelated prompts.

### 2. Efficacy

The edit must actually do the thing. Measured on the original prompt **and on
paraphrases the base model already answered the same way**. A paraphrase set the
base model fails is not a generalisation test, it is noise.

### 3. Reversibility

You must be able to undo it, exactly. Machaon copies the original bytes before
applying; `rollback()` restores them bit for bit.

## What an edit is

- A weight change. There is no optimiser over model parameters: the solve is
  closed-form or a gradient over one value vector.
- Persistent. It survives context changes, system prompts and restarts.
- Measured. Every edit spends a budget: `‖ΔW‖_F / ‖W‖_F`.

## The budget

```
rel(ΔW) = ‖ΔW‖_F / ‖W‖_F
```

Empirical starting points:

| edit | budget |
|---|---|
| factual, `down_proj` / `c_proj` | 0.01 – 0.03 |
| behavioural, `o_proj` | 0.05 – 0.15 |
| ablation | not clamped; measure KL instead |

Clamping is deliberate: it breaks the exact target guarantee so that the edit
spends no more of the model than you authorised.
