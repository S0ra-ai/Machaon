# FAQ

**Is this fine-tuning?**
No. There is no optimiser over model weights and no training data. An edit is a
closed-form or low-rank delta applied once. The only gradient anywhere is over a
single value vector during a ROME solve.

**Does it work on models other than GPT-2?**
The runtime discovers layers for Llama/Qwen/Mistral/Gemma (`model.layers`),
GPT-2/GPT-J (`transformer.h`), GPT-NeoX (`gpt_neox.layers`), OPT and MPT. Both
weight orientations are handled. Real-checkpoint evidence currently stops at
GPT-2 124M; 0.5B and up are the next rungs.

**Why not just use LoRA?**
LoRA requires a training loop and produces an adapter you must ship. This
produces a delta you can inspect, diff, and roll back — and it can tell you what
it did *not* change.

**Can I edit a 70B model?**
Not with this design. Reading one tensor at a time is fine, but the value solve
needs a forward pass and gradients through the remaining layers. The stated
target is up to 8B.

**Is a rank-one edit enough?**
For one fact, often. For a behaviour, low-rank plus a direction (`bake_direction`
or `ablate_output`) is usually better. For mass edits you want MEMIT-style batch
solving, which is not implemented.

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
that only edits items the model gets wrong has almost no headroom on stronger
models. Injection works at both ends.

**Why float32 instead of bf16?**
On CPU, PyTorch's bf16 backward is pathologically slow: 4.66 s/step versus
0.18 s/step in float32 on the same 0.5B model. The delta is computed in float64
either way; the model dtype only affects the forward/backward used to solve the
value.

**Why does the layer matter so much?**
An edit at layer 12 has to survive twelve more blocks. On Qwen2.5-0.5B the same
edit cost rel. Δ 0.150 at layer 12 and 0.0043 at layer 20 — a 35× difference.

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
