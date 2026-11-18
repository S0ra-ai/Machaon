# Weight programs

A weight program is a small text file that describes a complete, reviewable
weight-space operation:

- which checkpoint to start from;
- which facts the model should learn;
- which prompts must stay unchanged;
- how much of the model may change; and
- whether to reload the saved checkpoint and verify it.

It is the shareable form of the same loop used by `machaon first`,
`machaon teach-many`, and the browser console. Programs do not introduce a
second edit engine: they resolve paths and policy, call the teaching chain, and
then check the bytes that were actually written.

## A complete program

Save this as `hello.machaon`:

```text
version: 1
name: Hello, weight space

model: ./models/SmolLM2-135M
fetch: smol135m
output: ./models/hello-weight-space

budget: 0.10
max_rel_norm: 0.02
verify: true

teach: The Eiffel Tower is in the city of -> Rome
teach: The capital of France is -> Paris

protect: The capital of Germany is
protect: Water freezes at a temperature of
```

Check the resolved contract without downloading or loading a model:

```bash
machaon program hello.machaon --plan
```

Run the program:

```bash
machaon program hello.machaon
```

The model is downloaded only when `model` is missing and `fetch` names a
starter. Each fact gets its own search and local acceptance checks. The chain
keeps the last globally safe checkpoint at `<output>/final`, writes its own
`report.json`, and writes the program-level result to
`<output>/program-report.json`.

After saving, `verify: true` reloads that checkpoint and checks every fact
again. A program that saved a checkpoint but fails this reload is reported as
`verification_failed`, and exits non-zero. Use `--no-verify` only when you
intend to verify the bytes separately.

The local Web console exposes the same contract in its **Run a weight program**
panel: paste the program, set the folder used to resolve relative paths, and
choose **Check program** or **Run program**. The browser path still passes the
existing filesystem allowlist.

## Fields

Only `model`, `output`, and at least one `teach` are required.

| field | default | meaning |
|---|---|---|
| `model` | required | checkpoint directory; relative paths resolve from the program file |
| `output` | required | new checkpoint container; an existing non-empty directory is refused |
| `fetch` | none | starter to download if `model` is missing: `smol135m`, `qwen05b`, or `gpt2` |
| `teach` | required, repeatable | one teaching sentence, or a JSON `prompt` / `answer` object |
| `protect` | none, repeatable | unrelated prompts used for the baseline, projection, and panel |
| `budget` | `0.10` | cumulative relative-norm budget across all facts |
| `max_rel_norm` | `0.02` | per-fact relative-norm cap |
| `max_kl` | `0.05` | KL p95 limit on the protected panel |
| `min_panel_agreement` | `0.875` | fraction of protected prompts that must keep top-1 |
| `layers` | automatic depth window | `18,20,22` or `18-22` |
| `solver` | `direction` | `direction` or `gradient` |
| `dtype` | `float32` | `auto`, `float32`, `float16`, or `bfloat16` |
| `quantize` | none | `int4` or `int8`; requires `dtype: auto` |
| `device` | runtime default | `cpu`, `mps`, `cuda`, or a runtime-specific value |
| `verify` | `true` | reload the written checkpoint and check every fact |
| `continue_on_reject` | `false` | record a rejected fact and continue |
| `keep_steps` | `false` | keep every step checkpoint, not only the safe final |

Comments start with `#`. Blank lines are ignored. Each field uses `key: value`
or `key = value`; `teach` and `protect` may appear more than once.

## Relative paths

Relative `model` and `output` paths are resolved from the directory containing
the program file. This is what makes a program portable: copying the program
and its `models/` folder to another machine preserves the contract.

`--out DIR` overrides only the output for one run. The program hash and the
original declared output remain in the report, so an override cannot silently
rewrite the shared contract:

```bash
machaon program hello.machaon --out ./models/experiment-2
```

## JSON form

The same contract can be machine-generated as JSON. A fact may be a sentence,
`{"say": "..."}`, or an explicit `{"prompt": "...", "answer": "..."}` object:

```json
{
  "version": 1,
  "model": "./models/SmolLM2-135M",
  "fetch": "smol135m",
  "output": "./models/hello-weight-space",
  "facts": [
    "The Eiffel Tower is in the city of -> Rome",
    {"prompt": "The capital of France is", "answer": " Paris"}
  ],
  "protect": ["Two plus two equals"],
  "verify": true
}
```

Both forms produce the same `program-report.json`, including the program
SHA-256, the resolved policy, every teaching step, and the verification result.
That file is the audit record to keep when a program is shared or reviewed.

## What this does not claim

The local checks prove that the declared facts work and that the protected
panel stayed within its limits. They do not replace a benchmark run. A
program report is evidence about this edit, not a general claim about the
model.
