# Null-space projection

This is the one idea the project is built on.

## The problem

A rank-one edit that maps key `k` to value `v` is the minimum-norm solution to a
single constraint:

```
ΔW = (v − W k) kᵀ / (kᵀ k)
```

It is exact for `k` and says nothing about any other input. In practice it moves
neighbouring keys enough to damage them.

## The fix

Let `K = [k₁ … kₙ]ᵀ` be keys you refuse to change, and `Q` an orthonormal basis of
`span(K)`. Define the projector `P = I − Q Qᵀ` and write the edit along the
projected key instead:

```
k⊥  = P k
ΔW  = (v − W k) k⊥ᵀ / (k⊥ᵀ k)
```

Now:

- **target**: `ΔW k = v − W k`, so `W' k = v`. Exact.
- **protection**: `ΔW kᵢ = 0` for every protected key. Exact, to machine
  precision.

Note the numerator uses the **original** `k`. Using `k⊥` in both places is a
common bug: it lands the value but spends more norm than necessary.

## Implementation notes

Machaon never materialises the `d_in × d_in` projector in the hot path:

```python
Q = nullspace_basis(preserve)          # (d_in, r)
k_perp = project_complement(k, Q)      # k - Q Qᵀ k, O(d_in · r)
```

At `d_in = 4096` with `r = 32`, that is 0.5 MB of basis instead of a 128 MB
dense matrix.

## When it fails

**The target key lies in the protected subspace.** Then `k⊥ ≈ 0` and no edit can
satisfy both constraints. Machaon raises rather than silently degrading:

```
ValueError: target key lies in the preserved subspace: the edit cannot be made
without changing protected behaviour
```

This is a signal, not a bug: either shrink the protected set or pick a different
key.

## Scope of the guarantee

The guarantee is **first-order and exact on the protected set**: `ΔW kᵢ = 0`
holds to machine precision for the vectors you supply. Nearby inputs are
preserved to first order, and the post-hoc checks measure the remainder:
[`verify`](../reference/api.md) on the keys, KL on unrelated prompts.

Treat this as a scalpel with a known blade width, not as a proof that nothing
else moved.
