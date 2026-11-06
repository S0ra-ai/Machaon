# FAQ

**Is this fine-tuning?**
No. There is no optimiser over model weights and no training data. An edit is a
closed-form or low-rank delta applied once. The only gradient anywhere is over a
single value vector during a ROME solve.

**Does it work on models other than GPT-2?**
The runtime discovers layers for Llama/Qwen/Mistral/Gemma (`model.layers`),
GPT-2/GPT-J (`transformer.h`), GPT-NeoX (`gpt_neox.layers`), OPT and MPT. Both
weight orientations are handled. Real-checkpoint evidence covers GPT-2 124M,
Qwen2.5 0.5B / 1.5B / 3B / 7B and Llama-3.1-8B, with the same benchmark panel
at every rung.

**Why not just use LoRA?**
Measured, at equal steps and the same layer: a rank-8 adapter reaches
rewrite 1.000 / rephrase 1.000 but costs 44% perplexity and 25 MMLU points; the
rank-one edit reaches 1.000 / 0.976 for 4% perplexity and no measurable loss.
LoRA is more expressive; it is also unconstrained. See
[R9](RESULTS.md#r9-equal-budget-lora-baseline).

**Can I edit a 70B model?**
Machaon's measured range is through 8B, where int8/int4 weights plus a densified
local window keep the solve inside 10.2 GB. Beyond that, checkpoint-level
operations stream one tensor at a time, and the evaluation path scales with the
memory you give it. The solver cost is the window after the edit, not the whole
model.

**Is a rank-one edit enough?**
For one fact, often. For a behaviour, low-rank plus a direction (`bake_direction`
or `ablate_output`) is usually better. For mass edits, MEMIT-style batch solving
is the next primitive on the [roadmap](roadmap.md); today, compose per-fact
edits under one `EditBudget`, or use a low-rank direction.

**Why does my edit succeed but paraphrases fail?**
The key was tied to one token sequence. Capture keys from several phrasings and
average them; the rephrase set must also be one the base model already answers
correctly, or you are measuring noise.

**Why does `rank1_constrained` raise "preserved subspace"?**
The target key is inside the span of your protected keys. The two constraints are
incompatible. Shrink the protected set, or choose a different key.

**Why is `kl_p95` reported and not the mean?**
One damaged prompt matters more than a hundred untouched ones. The mean hides it.

**Why target injection instead of fixing wrong answers?**
Because Qwen2.5-0.5B answers 299 of 300 zsRE first tokens correctly. A protocol
that only edits items the model gets wrong leaves little measurement headroom
on stronger models. Injection keeps the panel informative at both ends of the
scale range.

**Why float32 instead of bf16?**
On CPU, PyTorch's bf16 backward is pathologically slow: 4.66 s/step versus
0.18 s/step in float32 on the same 0.5B model. The delta is computed in float64
either way; the model dtype only affects the forward/backward used to solve the
value.

**Why does the layer matter so much?**
An edit at layer 12 has to survive twelve more blocks. On Qwen2.5-0.5B the same
edit cost rel. Δ 0.150 at layer 12 and 0.0043 at layer 20 — a 35× difference.

**Hugging Face is blocked where I am.**
Use `python scripts/fetch_model.py --repo ... --dest ...`, which falls back to
ModelScope, or set `HF_ENDPOINT=https://hf-mirror.com`. The fetch script also
verifies safetensors headers after transfer.

**Why is repetition reported relative to baseline?**
Because GPT-2 loops under greedy decoding before any edit is applied. The
relative measure isolates the change produced by the edit from the base
model's decoding behaviour.

**How many edits can I stack?**
Measure it. On GPT-2, 32 rank-one edits (cumulative Δ 0.255) cost 19%
perplexity; 8 cost 5%. `EditBudget(max_total=0.10)` refuses around edit #10 and
is the recommended default.

**Why does the gate fail when a metric is missing?**
Because otherwise the easiest way to pass is to stop measuring. Fail-closed is a
design choice, not an oversight.

**Can I use the OpenWSP format without OpenWSP?**
Yes. `machaon.interop.openwsp` is a standalone implementation, conformance-tested
against a fixture produced by the reference implementation.

**Does this remove safety alignment?**
Direction ablation can, in one line. That is why the project ships refusal rules,
an append-only edit record and a gate suite. See
[SECURITY.md](https://github.com/S0ra-ai/Machaon/blob/main/SECURITY.md).
