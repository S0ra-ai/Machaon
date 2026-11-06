# The maths, and how it fails

## 1. Rank-one editing (ROME)

We want a weight `W ∈ R^{d_out × d_in}` to satisfy `W′ k = v` while changing as
little as possible. The minimum-Frobenius-norm solution to that single linear
constraint is rank one:

```
ΔW = (v − W k) kᵀ / (kᵀ k)
```

Check: `ΔW k = (v − Wk)(kᵀk)/(kᵀk) = v − Wk`, so `W′ k = v`. ∎

This is exact for the key and says **nothing** about any other input. That is
why naive ROME edits damage neighbours.

## 2. Null-space constrained editing

Let `K = [k₁ … kₙ]ᵀ` be keys you refuse to change, and let `Q` be an orthonormal
basis of `span(K)` (from the SVD of `K`). Define

```
P  = I − Q Qᵀ
k⊥ = P k
```

`k⊥` is orthogonal to every protected key. Use it as the write direction:

```
ΔW = (v − W k) k⊥ᵀ / (k⊥ᵀ k)
```

Two properties:

* **Target** — `ΔW k = (v − Wk)(k⊥ᵀk)/(k⊥ᵀk) = v − Wk`, so `W′ k = v`. Exact.
* **Protection** — `ΔW kᵢ = (v − Wk)(k⊥ᵀ kᵢ)/(k⊥ᵀk) = 0` for every `i`. Exact.

Note that the numerator uses the **original** `k`, not `k⊥`. Using `k⊥` in both
places is a common bug: it lands the value but spends more norm than necessary.

Implementation detail: we never materialise the `d_in × d_in` projector. The
basis form costs `O(d_in · r)` to apply, which matters at `d_in = 4096`, where a
dense projector would be 128 MB per layer.

## 3. Direction ablation

To remove the component of a module's output along a unit direction `d`:

```
W′ = (I − d dᵀ) W
```

Applied to every matrix that writes into the residual stream, this removes a
behaviour (a refusal direction, a style direction) without any data. One line,
zero gradients. This is also why the capability is dangerous — see the safety
note below.

## 4. Direction injection

Inference-time steering adds `α d` to a hidden state. Because the residual
stream is additive, that is exactly equivalent to adding a rank-one term to the
module that writes there:

```
ΔW = α d aᵀ / (aᵀ a)
```

for a representative input activation `a`. Now `W′ a = W a + α d` — the steering
vector is baked into the weights and needs no runtime hook.

## 5. The budget

```
rel(ΔW) = ‖ΔW‖_F / ‖W‖_F
```

This is the only safety dial that generalises across layers. Clamping to
`max_rel_norm` scales the delta down, which **breaks the exact target
guarantee** — deliberately. A capped edit stops early rather than spending more
of the model than you authorised.

Empirical starting points:

| edit type | budget |
|---|---|
| factual, `down_proj` | 0.01 – 0.03 |
| behavioural, `o_proj` | 0.05 – 0.15 |
| ablation | not clamped; measure KL instead |

## How it fails

| symptom | cause | fix |
|---|---|---|
| exact prompt works, paraphrase fails | key tied to one token sequence | average over 5–20 phrasings |
| short answer right, long text incoherent | edit broke downstream computation | shallower layer, lower budget, stronger projection |
| capability drops | layer too central | move 2–4 layers earlier |
| unrelated prompts drift | no null-space constraint | add a protected key set |
| nothing changes in bf16 | delta lost to rounding | compute in float64, merge back after |
| cumulative edits collapse the model | deltas accumulate | cap `Σ‖ΔWᵢ‖/‖W‖` at ~0.1, then recompute |

## Where the guarantees stop

Everything above is **first-order and local**. `ΔW kᵢ = 0` holds for the exact
vectors in your protected set; nearby inputs are only approximately preserved.
Non-linearities downstream amplify small errors. More complex function changes
are the job of low-rank, directional or multi-edit compositions.

Treat these formulas as a scalpel with a known blade width, not as a proof that
nothing else moved. That is what the gates are for.

## Safety

Direction ablation can remove safety alignment in a single line. Any deployment
of this library should keep an append-only audit log of every edit, refuse
edits whose intent is to remove refusal behaviour, and never overwrite a
checkpoint in place.
