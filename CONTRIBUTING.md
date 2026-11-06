# Contributing

Thanks for looking. This project is small and opinionated; the fastest way in is
to reproduce something and tell us what broke.

## Setup

```bash
git clone https://github.com/S0ra-ai/Machaon.git
cd Machaon
uv venv && uv pip install -e ".[dev,docs]"
pre-commit install
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
make test-cov            # branch coverage, 78% floor in pyproject
make docs                # local docs site; CI builds with --strict
```

Everything that CI runs is runnable offline. If `./scripts/ci.sh` passes locally,
CI will pass.

The full matrix is Python 3.9–3.13. The optional `hf` extra adds a second job
that runs the runtime, adapter and evaluation tests with torch installed.

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
- the evidence behind any new number, and where the measured claim ends.

## Documentation

Documentation is part of the public surface. A page that names a function,
threshold or number must match the code and the measured run. When you change
one, update the English page and its `.zh.md`, `.ja.md` and `.ko.md`
counterparts. `mkdocs build --strict` is part of CI, and
`tests/test_metadata.py` checks the four-language coverage, test-count badges,
package metadata and the documentation tone.

## Release process

1. Move `[Unreleased]` entries in `CHANGELOG.md` under the new version.
2. Bump `version` in `pyproject.toml`, `machaon/__init__.py` and
   `CITATION.cff`; `tests/test_metadata.py` enforces that they agree.
3. Push a `vX.Y.Z` tag. `release.yml` builds sdist + wheel, checks the tag
   against the package version, and publishes a GitHub release. Set the
   repository variable `PUBLISH_TO_PYPI=true` to publish through PyPI trusted
   publishing.

## Reporting a problem

Open an issue with the smallest reproduction you can manage, including the model
name and revision if a checkpoint is involved.
