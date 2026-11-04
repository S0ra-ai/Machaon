# Contributing

Thanks for looking. This project is small and opinionated; the fastest way in is
to reproduce something and tell us what broke.

## Setup

```bash
git clone https://github.com/S0ra-ai/Machaon.git
cd Machaon
uv venv && uv pip install -e ".[dev]"
```

PyTorch and transformers are optional:

```bash
uv pip install -e ".[hf]"      # real forward passes
uv pip install ruff            # lint
```

## Run the checks

```bash
./scripts/ci.sh          # tests, demo, spec validation, gate behaviour
make lint                # ruff
```

Everything that CI runs is runnable offline. If `./scripts/ci.sh` passes locally,
CI will pass.

## What we want help with, in order

1. **A real result.** Any edit on a public model under 8B with before/after
   numbers that survive the gates. This is worth more than any refactor.
2. **More architectures.** `machaon/runtime/hf.py` has a `LAYER_PATHS` table.
   Adding a family means adding a path and a test, not changing the metrics.
3. **More edit primitives** with a derivable guarantee (like the null-space
   projection) and a test that shows the guarantee holds.
4. **Evaluation suites.** Long-form coherence and ripple effects are the two
   places existing work usually falls over.

## Ground rules

- **No silent failure.** New acceptance checks must fail closed: a missing
  measurement is a failure, not a pass.
- **No unverifiable claims.** If the README says something is implemented,
  there is a test that fails when it is removed.
- **Edits are recorded as data.** Prefer a declarative spec over a Python call
  that does the work inline.
- **Keep the core dependency-light.** NumPy is the floor. Torch is an adapter.

## Commit messages

`area: imperative summary` — for example `linalg: guard degenerate key geometry`.
Explain *why* in the body when the change is not obvious.

## Pull requests

Keep them small. One idea per PR. Include:
- what changes,
- how you verified it (command + output),
- what it does *not* claim.

## Reporting a problem

Open an issue with the smallest reproduction you can manage, including the model
name and revision if a checkpoint is involved.
