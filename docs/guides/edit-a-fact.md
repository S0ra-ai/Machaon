# Edit a fact

The full procedure, as implemented in `machaon.procedures.rome`.

## The six steps

### 1 · Choose the matrix

Facts usually live in the MLP block of a middle layer. Start around 60–70% of
depth and move earlier if the edit damages capability.

```python
from machaon.runtime import HFModel

runtime = HFModel(model, tokenizer)
name = runtime.edit_target(layer=8)          # transformer.h.8.mlp.c_proj.weight
```

`edit_target` translates names across families (`down_proj` ⇄ `c_proj`) and
normalises weight orientation, so the maths always sees `(d_out, d_in)`.

### 2 · Capture the key

The key is what the matrix *actually* sees, not the residual stream before the
final layer norm.

```python
ids = runtime.encode(["The Eiffel Tower is located in the city of"])[0]
key = runtime.capture_module_input("transformer.h.8.mlp.c_proj", ids)[0]
```

Average over several phrasings if the fact is not tied to one template.

### 3 · Build the protected set

```python
neutral = ["The capital of Germany is", "Water freezes at a temperature of", ...]
preserve = np.stack([runtime.capture_module_input(module, runtime.encode([p])[0])[0]
                     for p in neutral])
```

These are the keys that must not move. Eight to thirty is usually enough; more
protection costs efficacy.

### 4 · Solve the value

The value is the output the matrix should produce for that key. ROME optimises
it with a short gradient run — over a **single vector**, never over the model:

```python
result = rome_edit(runtime, prompt, target, layer=8,
                   preserve_prompts=neutral, steps=250)
```

### 5 · Apply with a budget

```python
W_new, delta = rank1_constrained(W, key, value, preserve=preserve,
                                 max_rel_norm=0.02)
```

The budget is a hard cap on `‖ΔW‖/‖W‖`. If the edit needs more, it under-delivers
rather than overspending.

### 6 · Verify

```python
from machaon.verify import check_protected
check_protected(W, delta, preserve)     # max_drift should be ~1e-16
```

## Run it end to end

```bash
export HF_ENDPOINT=https://hf-mirror.com
python examples/gpt2_rome.py
```

```
model      : gpt2 (12 layers)
edit       : transformer.h.8.mlp.c_proj.weight
solved p(t): 0.9968 after 250 steps
delta      : rel 0.0153
  rewrite_acc  1
  paraphrase_acc 1
  kl_p95       0.000614724
  ppl_ratio    1.00079
PASS (5 gates)
```

## Failure modes

| symptom | cause | fix |
|---|---|---|
| exact prompt works, paraphrase fails | key tied to one template | average over phrasings |
| capability drops | layer too central | move 2–4 layers earlier |
| nothing changes in bf16 | delta lost to rounding | compute in float64 |
| unrelated prompts drift | no protected set | add one |
| edit refuses with "preserved subspace" | target key is inside the protected span | shrink the set |
