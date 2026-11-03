# Related work

The idea sits at the intersection of five communities. None of them ships the
whole loop.

## Parameter-level knowledge editing

| work | note |
|---|---|
| [ROME](https://arxiv.org/abs/2202.05262) | rank-one factual edits; the origin of this line |
| [MEMIT](https://arxiv.org/abs/2210.07229) | batch editing thousands of facts |
| [MEND](https://arxiv.org/abs/2110.11309) | hypernetwork that predicts edits |
| [SERAC](https://arxiv.org/abs/2206.06520) | external memory instead of weights |
| [Survey](https://arxiv.org/abs/2310.19704) | best entry point to the literature |
| [EasyEdit](https://github.com/zjunlp/EasyEdit) | the closest existing framework (2.9k★), batch-script oriented |

## Training-free weight surgery

| work | note |
|---|---|
| [Model Surgery](https://arxiv.org/abs/2407.08770) | very few parameters change behaviour |
| [SCALPEL](https://arxiv.org/abs/2601.07411) | low-rank parameter editing for capability ablation |
| [ThinkEdit](https://arxiv.org/abs/2503.22048) | weight edits change reasoning length |
| [Resisting Harmful Instructions through Training-Free Weight Editing](https://arxiv.org/abs/2610.21189) | proof this is practical — and dangerous |
| [TARS](https://arxiv.org/abs/2412.10257) | targeted angular reversal for knowledge removal |

## Representation and activation editing

| work | note |
|---|---|
| [Refusal is mediated by a single direction](https://arxiv.org/abs/2406.11717) | the "one direction, one behaviour" result |
| [RepE](https://arxiv.org/abs/2310.01405) | representation engineering |
| [CAA](https://arxiv.org/abs/2312.06681) | contrastive activation addition |
| [ActAdd](https://arxiv.org/abs/2306.03341) | activation engineering |
| [SAELens](https://github.com/decoderesearch/SAELens) | sparse-autoencoder features as edit targets |

## Weight-space algebra

| work | note |
|---|---|
| [Task Arithmetic](https://arxiv.org/abs/2212.04089) | editing models by adding task vectors |
| TIES / DARE | interference-aware merging |
| [mergekit](https://github.com/arcee-ai/mergekit) | the mature toolchain (7.4k★) |
| [Evolutionary model merge](https://github.com/SakanaAI/evolutionary-model-merge) | search over merge recipes |

## LLM-driven optimisation

| work | note |
|---|---|
| [OPRO](https://arxiv.org/abs/2309.03409) | LLMs as optimisers |
| [TextGrad](https://arxiv.org/abs/2406.07496) | textual gradients |
| [SEAL](https://arxiv.org/abs/2506.10943) | a model generates its own update directives |
| [Foundation models understand weights](https://arxiv.org/abs/2503.00838) | supports "an LLM can reason about weight space" |
| [LLM as a Tool, Not an Agent](https://arxiv.org/abs/2604.16555) | the sharpest counter-argument to agentic editing |
| [Agentic NAS](https://arxiv.org/abs/2607.07984) | the opposing camp |

## The two papers to read before building

1. [Revisiting Parameter-Based Knowledge Editing: Theoretical Limits and Empirical Evidence](https://arxiv.org/abs/2606.00570) —
   argues parameter editing has fundamental ceilings.
2. [Long-form evaluation of model editing](https://arxiv.org/abs/2402.09394) —
   edits that pass short QA collapse in long generation.

Both are load-bearing for the gate design in `machaon.hygieia`.

## The gap

1. No agent-operable edit action space — existing work is a Python API or a script.
2. No transaction layer — nothing snapshots, verifies, and rolls back.
3. Evaluation is not a first-class citizen.
4. No system lets the LLM *plan the surgery*: diagnose → choose site → declare
   constraints → hand off to a closed-form solver.
