"""Content-addressed WSP evidence bundles.

The WSP decision is only useful if it can be replayed and audited later.  This
module wraps the JSON report produced by :mod:`machaon.wsp` in a small,
versioned document with a canonical SHA-256 address.

The bundle is intentionally plain JSON: it can be committed, diffed, attached
to an edit manifest, or read by an agent without loading a model.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

FORMAT = "machaon.wsp.evidence"
FORMAT_VERSION = 1
FILE_NAME = "evidence.json"

__all__ = [
    "FORMAT",
    "FORMAT_VERSION",
    "EvidenceError",
    "evidence_sha256",
    "write_evidence",
    "read_evidence",
    "is_wsp_evidence",
]


class EvidenceError(ValueError):
    """A WSP evidence bundle is malformed, corrupted, or incompatible."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EvidenceError("evidence is not JSON-serialisable: %s" % exc) from exc


def evidence_sha256(document: Mapping[str, Any]) -> str:
    """Address the document excluding the address field itself."""
    body = {k: v for k, v in document.items() if k != "evidence_sha256"}
    return hashlib.sha256(_canonical(body)).hexdigest()


def _target_path(path: str | Path) -> Path:
    target = Path(path)
    if target.suffix.lower() == ".json":
        return target
    return target / FILE_NAME


def write_evidence(
    path: str | Path,
    report: Mapping[str, Any],
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Write one canonical evidence document and return it.

    ``path`` may be a JSON file or a directory; a directory receives
    ``evidence.json``.  The document is published atomically.
    """
    if not isinstance(report, Mapping):
        raise EvidenceError("report must be an object")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise EvidenceError("metadata must be an object")
    document: Dict[str, Any] = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "report": dict(report),
        "metadata": dict(metadata or {}),
    }
    document["evidence_sha256"] = evidence_sha256(document)

    target = _target_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=target.parent, suffix=".json", delete=False, encoding="utf-8"
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(document, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    try:
        os.replace(temp_path, target)
    finally:
        temp_path.unlink(missing_ok=True)
    return document


def read_evidence(path: str | Path) -> Dict[str, Any]:
    """Read and fully verify an evidence document."""
    target = _target_path(path)
    try:
        document = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EvidenceError("cannot read evidence: %s" % exc) from exc
    if not isinstance(document, dict):
        raise EvidenceError("evidence must be an object")
    if document.get("format") != FORMAT:
        raise EvidenceError("unsupported evidence format: %r" % document.get("format"))
    if document.get("format_version") != FORMAT_VERSION:
        raise EvidenceError(
            "unsupported evidence version: %r" % document.get("format_version")
        )
    if "report" not in document or not isinstance(document["report"], dict):
        raise EvidenceError("evidence needs a report object")
    if document.get("evidence_sha256") != evidence_sha256(document):
        raise EvidenceError("evidence address mismatch")
    return document


def is_wsp_evidence(path: str | Path) -> bool:
    """True when ``path`` is a valid, address-verified WSP evidence bundle."""
    try:
        read_evidence(path)
    except EvidenceError:
        return False
    return True
