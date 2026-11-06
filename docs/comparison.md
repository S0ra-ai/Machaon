# How Machaon fits

Different jobs want different tools. This page is the short map: what each
approach changes, what it can undo, and where Machaon adds a contract.

| approach | unit of change | undo | locality evidence | best fit |
|---|---|---|---|---|
| **Machaon** | one targeted delta, content-addressed | bit-exact rollback | protected-key drift + KL + gates | auditable single edits, agent workflows |
| Fine-tuning | all weights, via gradients | checkpoint restore | benchmark suite | broad capability or style change |
| LoRA / PEFT | adapter weights | remove the adapter | benchmark suite | parameter-efficient adaptation |
| ROME / MEMIT | rank-one or batch delta | none by default | paper-specific probes | research on factual editing |
| Activation steering | runtime hook | remove the hook | probe prompts | inference-time control |
| mergekit / task arithmetic | whole-model merge | keep both parents | merge evaluation | combining trained models |
| pruning / rewiring | structure | keep the original checkpoint | benchmark suite | size or latency reduction |

Machaon is deliberately narrow: it owns the **targeted edit** and the
**acceptance contract** around it. The other rows remain one call away, and
their artefacts can enter the same audit:

- a mergekit or task-arithmetic delta can be checked with `machaon.verify` and
  applied with `machaon.tools.apply_delta`;
- an OpenWSP edit from another tool can be read, hash-checked and gated;
- a LoRA adapter gives a like-for-like baseline, measured at equal steps and
  the same layer in [R9](RESULTS.md#r9-equal-budget-lora-baseline);
- a steering direction can be baked into weights with `bake_direction`, then
  audited like any other delta.

## What Machaon adds to a parameter edit

1. **A declared constraint.** Protected keys are part of the edit, not a
   post-hoc hope.
2. **A budget.** `Σ‖ΔWᵢ‖/‖W‖` is charged at commit and refuses overspend.
3. **A transaction.** Every edit is reversible byte for byte.
4. **A verdict.** Eight fail-closed gates, with missing metrics counted as
   failures.
5. **A portable record.** OpenWSP v1, byte-identical, hash-checked on import.
6. **An agent surface.** Twelve MCP tools, file-level and model-level.

## Choosing in one line

- Change one fact or one behaviour, and you need evidence: **Machaon**.
- Change the whole model's style or capability: **fine-tuning**.
- Add a small trainable adapter: **LoRA**.
- Combine two trained models: **mergekit**.
- Steer at inference time without writing weights: **activation steering**.
