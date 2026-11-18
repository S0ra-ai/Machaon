"""Verify the link between an OpenWSP edit and its WSP evidence."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np

from .evidence import EvidenceError, read_evidence
from .interop import openwsp

__all__ = ["BindingReport", "verify_binding"]


@dataclass(frozen=True)
class BindingReport:
    passed: bool
    edit_sha256: str | None = None
    evidence_sha256: str | None = None
    label: str | None = None
    errors: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "passed": self.passed,
            "edit_sha256": self.edit_sha256,
            "evidence_sha256": self.evidence_sha256,
            "label": self.label,
            "errors": list(self.errors),
        }


def _payload_matches(edit_dir: Path, manifest) -> bool:
    try:
        with np.load(edit_dir / "edit.npz", allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
    except (OSError, ValueError):
        return False
    return manifest.get("payload_sha256") == openwsp.payload_sha256(arrays)


def verify_binding(edit_dir: str | Path, evidence_dir: str | Path) -> BindingReport:
    """Check edit address, evidence address and manifest payload together.

    This does not load the base checkpoint.  It proves the two artefacts are
    bound to each other and that each internal address matches its bytes; a
    caller still needs ``openwsp.read_edit`` with the base checkpoint for the
    stronger tensor-level audit.
    """
    edit_dir = Path(edit_dir)
    errors: List[str] = []
    try:
        evidence = read_evidence(evidence_dir)
    except EvidenceError as exc:
        return BindingReport(False, errors=[str(exc)])

    try:
        manifest = json.loads((edit_dir / "manifest.json").read_text())
    except (OSError, ValueError) as exc:
        return BindingReport(
            False,
            evidence_sha256=evidence.get("evidence_sha256"),
            label=evidence.get("report", {}).get("label"),
            errors=["cannot read OpenWSP manifest: %s" % exc],
        )
    if not isinstance(manifest, dict):
        return BindingReport(False, errors=["OpenWSP manifest must be an object"])
    if manifest.get("format") != openwsp.FORMAT:
        errors.append("unsupported OpenWSP format")
    if manifest.get("format_version") != openwsp.FORMAT_VERSION:
        errors.append("unsupported OpenWSP format version")

    edit_sha = openwsp.edit_sha256(manifest)
    if manifest.get("edit_sha256") != edit_sha:
        errors.append("OpenWSP edit address mismatch")
    if not _payload_matches(edit_dir, manifest):
        errors.append("OpenWSP payload address mismatch")

    binding = manifest.get("machaon") or {}
    bound_evidence = binding.get("evidence_sha256")
    if not bound_evidence:
        errors.append("manifest has no machaon.evidence_sha256 binding")
    elif bound_evidence != evidence.get("evidence_sha256"):
        errors.append("manifest evidence address does not match evidence bundle")
    bound_format = binding.get("evidence_format")
    if bound_format and bound_format != evidence.get("format"):
        errors.append("manifest evidence format does not match evidence bundle")

    return BindingReport(
        passed=not errors,
        edit_sha256=edit_sha,
        evidence_sha256=evidence.get("evidence_sha256"),
        label=evidence.get("report", {}).get("label"),
        errors=errors,
    )
