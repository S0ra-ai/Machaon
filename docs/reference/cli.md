# CLI

```bash
python -m machaon.cli <command>
```

## `first`

The shortest path from zero. If `--model` does not exist, it downloads the
selected starter (SmolLM2-135M by default) with the same endpoint fallback and
header validation as `machaon fetch`; then it opens the checkpoint, teaches one
sentence, and saves the accepted edit.

```bash
machaon first
machaon first --say "The capital of France is -> Berlin"
machaon first --model ./models/Qwen2.5-0.5B --out ./models/qwen-first
```

The default fact is "The Eiffel Tower is in the city of -> Rome". The default
output is `<model>-machaon`. Use `--no-save` to keep the edit in memory only;
`--json` prints the full result. The command exits 0 when the fact was applied
or already true, 1 when the local checks rejected it, and 2 on malformed input
or a model error.

## `check`

Reloads a saved checkpoint and evaluates one prompt/answer pair without editing
anything. It tries the answer with and without the continuation space, reports
the variant that matches the model's next-token prediction, and exits 0 only
when the target token is reproduced.

```bash
machaon check --model ./models/qwen-first \
  --say "The Eiffel Tower is in the city of -> Rome"
machaon check --model ./models/qwen-first \
  --prompt "The Eiffel Tower is in the city of" --answer Rome --json
machaon check --model ./models/qwen-many/final --facts facts.txt
```

The default `--dtype auto` reads the stored checkpoint dtype. `--quantize`
requires `--dtype auto`. The JSON output includes `passed`, `accuracy`,
`answer_text` and the raw metrics; the human form prints `PASS` or `FAIL`.
`--facts` loads the list once and checks every entry, exiting 1 if any fact
fails.

## `fetch`

Downloads a safetensors checkpoint without importing torch or loading a weight
tensor. It lists the repository through Hugging Face, then `hf-mirror`, then
ModelScope, and validates every `.safetensors` header before accepting it.
Existing files are reused, so an interrupted download can resume.

```bash
machaon fetch --list
machaon fetch --starter qwen05b --dest ./models/Qwen2.5-0.5B
machaon fetch --starter smol135m --dest ./models/SmolLM2-135M
machaon fetch --repo Qwen/Qwen2.5-0.5B --dest ./models/qwen
```

`--endpoint huggingface|hf-mirror|modelscope` forces one endpoint. `--json`
prints the file list, shard count, tensor count and parameter count. A
repository with no `.safetensors` weights is refused rather than left as a
half-downloaded directory.

## `start`

Prints the shortest path to a first edit and reports which optional runtimes are
installed:

```bash
python -m machaon.cli start
python -m machaon.cli start --demo
```

## `web`

Launch the optional local browser console from the main CLI. It forwards the
same host, port, allowed-root, token and loopback safeguards as `machaon-web`;
install the `web` extra first.

```bash
pip install "machaon[web]"
machaon web --allowed-root ./models
machaon web --allowed-root ./models --port 8770 --no-browser
```

## `intent`

Compiles a plain-language teaching request into an exact `prompt` / `answer`
contract without loading a model. Exit code 2 means the text did not determine
one pair; the error lists accepted forms.

```bash
python -m machaon.cli intent "The Eiffel Tower is in the city of -> Rome"
python -m machaon.cli intent "教会模型：北京的首都是 → 北京" --json
```

Arrow and labelled forms are exact. Plain sentences are also accepted and are
marked `medium` confidence because the split point is an assumption. The
command prints the equivalent `machaon teach` command.

## `teach`

Runs the full teaching recipe: baseline, depth-window layer search, apply,
target/panel/KL checks, and rollback on failure.

```bash
python -m machaon.cli teach --model ./models/Qwen2.5-0.5B \
  --say "The Eiffel Tower is in the city of -> Rome"
```

The default output is a sibling directory named `<checkpoint>-machaon`. The
source checkpoint is never modified. Common controls:

| flag | meaning |
|---|---|
| `--say TEXT` | compile the request with `machaon intent` |
| `--prompt P --answer A` | supply the contract explicitly |
| `--layers 8,10,12` / `--layers 8-12` | choose candidate layers |
| `--max-layers 7` | default candidate count for the depth window |
| `--solver direction\|gradient` | value solver for each candidate |
| `--max-rel-norm 0.02` | per-edit relative-norm budget |
| `--budget 0.10` | cumulative session budget |
| `--max-kl 0.05` | KL p95 acceptance threshold |
| `--min-panel-agreement 0.875` | fraction of unrelated prompts that must keep top-1 |
| `--preserve-file FILE` | newline-separated baseline/projection prompts |
| `--corpus-file FILE` | newline-separated texts for an optional perplexity check |
| `--out DIR` | edited checkpoint directory (refused if it exists) |
| `--no-save` | keep the edit in memory only |
| `--dry-run` | search and report, write nothing |
| `--evidence-out DIR` | write a WSP evidence bundle with the unedited headroom |
| `--edit-out DIR` | export the applied delta as OpenWSP v1 |
| `--report-out FILE` | write the full recipe report JSON |
| `--json` | machine-readable output |

Exit codes: `0` applied / already true / dry run, `1` rejected and rolled back,
`2` malformed input or model error.

`--out`, `--evidence-out` and `--edit-out` are refused when the target already exists, so a run never overwrites an earlier artifact.

## `teach-many`

Teach a list of facts into one checkpoint chain. Each fact runs the full
single-fact recipe on a fresh session loaded from the previous accepted
checkpoint; the original checkpoint is never modified.

```bash
machaon teach-many --model ./models/Qwen2.5-0.5B \
  --facts facts.txt --out ./models/qwen-many
```

`facts.txt` contains one teaching sentence per line (the same grammar as
`machaon intent`) or a JSON / JSONL list of `{"say": ...}` or
`{"prompt": ..., "answer": ...}` objects. The command keeps the last globally
safe checkpoint at `<out>/final`, writes per-step evidence to
`<out>/report.json`, and exits 1 when a fact is rejected or the cumulative
drift check fails. `--continue-on-reject` records the failure and continues;
`--keep-steps` keeps every step rather than only the last safe one; in that
mode the final is the last `step-NNN` directory instead of `<out>/final`. The
cumulative `--budget` is enforced across the chain, not per fact.

## `program`

Plan or run a portable weight program. A program is a single text or JSON
file that declares the model, facts, protected prompts, budget and runtime
policy. Relative model and output paths resolve from the program file, so the
same file can be shared and reviewed.

```bash
machaon program examples/hello.machaon --plan
machaon program examples/hello.machaon
machaon program ./lesson.json --out ./models/experiment-2
```

`--plan` parses and prints the exact contract without downloading or loading a
model. A run delegates the edits to `teach-many`, writes
`<output>/program-report.json`, and by default reloads the saved checkpoint and
checks every fact. `--no-verify` skips that final reload; `--json` prints the
whole plan or run for another tool.

Exit codes are `0` for a verified/applied program (or a valid plan), `1` for a
rejected or verification-failed run, and `2` for a malformed program or model
error. See [Weight programs](../guides/weight-programs.md) for the format and
field table.

## `demo`

Runs the synthetic rank-one demo. No model, no network.

```bash
python -m machaon.cli demo
```

## `validate`

Validates an edit spec. Exit code 1 on an invalid spec.

```bash
python -m machaon.cli validate examples/edit_spec.json
# OK  rank1_constrained -> model.layers.14.mlp.down_proj (budget 0.020)
```

## `gate`

Applies the acceptance gates to a metrics JSON file. Exit code 1 when the edit
should not be committed — which makes it usable as a CI step.

```bash
python -m machaon.cli gate examples/metrics_pass.json    # exit 0
python -m machaon.cli gate examples/metrics_fail.json    # exit 1
```

```
FAIL
  paraphrase_acc = 0.41 violates >= 0.6
  kl_p95 = 0.113 violates <= 0.05
  ...
```

## `wsp`

Applies the WSP acceptance protocol to a JSON report. Exit code 0 only when the
edit is a gate-passing `repair` or `acquisition`; every other label exits 1.

```bash
python -m machaon.cli wsp examples/wsp_report.json
```

The report contains `base`, `candidate`, optional `controls`, optional
`headroom`, optional `mechanism`, optional `gates`, and optional `protocol`
objects. The full classification, control/headroom gaps and mechanism gate are
printed as JSON.

`--save DIR` writes a content-addressed evidence bundle; `--metadata FILE`
adds a JSON object to it.

```bash
python -m machaon.cli wsp examples/wsp_report.json --save evidence/
python -m machaon.cli wsp-verify evidence/
```

### WSP benchmark

```bash
python -m machaon.cli wsp-bench examples/wsp_benchmark.json
python -m machaon.cli wsp-bench measured.json --require-measured
```

Each case declares `failure_type` (`obstruction` or `capability_gap`) and
`expected_repairable`. The command prints per-case WSP evidence plus an
agreement/confusion summary, and exits `1` when any case disagrees with its
label. The shipped fixture is synthetic and must not be reported as a model
measurement. `--require-measured` exits `2` if any case is not measured.
`examples/wsp_measured_smoke.json` is a measured threshold-classifier reference
generated by `scripts/make_wsp_reference_case.py`; it validates the provenance
gate but is not an LLM capability claim.

### Planning

```bash
python -m machaon.cli plan examples/plan_obstruction.json
```

The input declares `failure_type` and either `layer_scores` or a causal-patching
`patch_report`. The output is an `edit`, `diagnose` or `do_not_edit` route. An
obstruction without a site above threshold is not turned into a guessed layer.

### Diagnosis

```bash
python -m machaon.cli diagnose examples/diagnose_obstruction.json
```

The input supplies `base`, optional `headroom`, optional `mechanism`, an
optional diagnosis `protocol`, and optional `plan` arguments. The output
contains the conservative failure type and, when requested, the corresponding
edit/diagnose/do-not-edit plan. Missing or contradictory evidence stays
`unknown`.

`wsp-verify` checks the format/version and recomputes the evidence SHA-256.

## `binding`

Verify that an OpenWSP edit and a WSP evidence bundle still point at each other:

```bash
python -m machaon.cli binding out/edit evidence/
```

Exit code 1 means the edit address, payload address or evidence binding failed.

## `inspect`

Lists tensors in a safetensors checkpoint.

```bash
python -m machaon.cli inspect /models/qwen --suffix down_proj.weight
python -m machaon.cli inspect /models/qwen --layer 3 --limit 5
```

## `--version`

```bash
python -m machaon.cli --version
```

## The three-command CI

```bash
./scripts/ci.sh
```

Runs the tests, the demo, spec validation, both gate cases, and lint.
