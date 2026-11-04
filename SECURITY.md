# Security

## Reporting

Open a private security advisory on GitHub, or email the maintainer listed in
`pyproject.toml`. Please do not open a public issue for a vulnerability.

## What counts as a vulnerability here

This library edits model weights. That gives it a genuinely dual-use surface, so
we treat the following as security issues rather than bugs:

- a path that removes or weakens safety behaviour without passing a gate;
- an edit that escapes the relative-norm budget;
- a checkpoint write that silently alters tensors outside the declared delta;
- an OpenWSP edit that is accepted despite a hash mismatch;
- a gate that passes when a required metric is missing.

## Design commitments

- Every applied edit is recorded; nothing is overwritten in place.
- Refusals run before acceptance, not after.
- Gates fail closed.
- The delta store is content-addressed and verifiable offline.

## Known limitation

`machaon.podalirius` executes forward hooks inside the process that owns the
model. The Python kernel in a future MCP server will not be an operating-system
sandbox. Do not expose it to untrusted input.
