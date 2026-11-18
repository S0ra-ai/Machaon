"""Operations an agent can call without loading a model.

These are plain functions: no transport, no globals, no model state. The MCP
server and the CLI are thin wrappers around them, and the tests exercise them
directly.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .binding import verify_binding
from .evidence import write_evidence
from .hygieia.gates import evaluate
from .intent import parse_intent as _parse_intent
from .interop import openwsp
from .planning import diagnose_failure as _diagnose_failure
from .planning import plan_surgery as _plan_surgery
from .refusals import check_tensor
from .verify import audit_edit
from .wsp import assess_report


def _checkpoint(path: str):
    """Imported lazily so this module works without the safetensors extra."""
    from .adapters import Checkpoint

    return Checkpoint(path)


__all__ = [
    "describe_checkpoint",
    "list_tensors",
    "audit_checkpoint_edit",
    "apply_delta",
    "gate_metrics",
    "wsp_audit",
    "save_wsp_evidence",
    "verify_wsp_binding",
    "plan_surgery",
    "diagnose_failure",
    "parse_intent",
]


def describe_checkpoint(path: str) -> Dict[str, Any]:
    """Tensor count, parameter count, dtype mix and shard count. Header only."""
    return _checkpoint(path).summary()


def list_tensors(path: str, contains: Optional[str] = None,
                 suffix: Optional[str] = None, layer: Optional[int] = None,
                 limit: int = 50) -> List[Dict[str, Any]]:
    """Bounded tensor listing; values are never returned."""
    checkpoint = _checkpoint(path)
    names = checkpoint.find(contains=contains, suffix=suffix, layer=layer)
    out = []
    for name in names[: max(1, int(limit))]:
        info = checkpoint.info(name)
        out.append({"name": name, "shape": list(info.shape), "dtype": info.dtype})
    return out


def audit_checkpoint_edit(base_path: str, edit_dir: str,
                          max_rel_norm: Optional[float] = None) -> Dict[str, Any]:
    """Verify an OpenWSP v1 edit against a base checkpoint, then report drift.

    This is the operation that does not require trusting the tool that produced
    the edit: hashes are checked against the base, and the relative change is
    recomputed independently.
    """
    manifest_path = Path(edit_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    names = [record["name"] for record in manifest.get("tensors", [])]
    if not names:
        return {"passed": True, "changed_tensors": 0, "tensors": {},
                "edit_sha256": manifest.get("edit_sha256")}

    checkpoint = _checkpoint(base_path)
    originals = {name: checkpoint.read(name) for name in names}
    verified, rebuilt = openwsp.read_edit(edit_dir, originals)
    deltas = openwsp.delta_from_edit(verified, rebuilt, originals)

    report = audit_edit(deltas, originals)
    worst = report["worst_relative_change"]
    within = max_rel_norm is None or worst <= max_rel_norm
    return {
        "passed": bool(within),
        "edit_sha256": verified.get("edit_sha256"),
        "machaon": verified.get("machaon"),
        "changed_tensors": int(verified.get("changed_tensors", len(deltas))),
        "worst_relative_change": worst,
        "max_rel_norm": max_rel_norm,
        "tensors": report["tensors"],
    }


def apply_delta(base_path: str, tensor: str, delta_path: str, out_dir: str,
                refuse_rescale: bool = True) -> Dict[str, Any]:
    """Apply one ``.npy`` delta to one tensor and write a new checkpoint."""
    checkpoint = _checkpoint(base_path)
    base = checkpoint.read(tensor)
    delta = np.load(delta_path, allow_pickle=False)
    if not np.isfinite(delta).all():
        raise ValueError("delta contains non-finite values")
    if delta.shape != base.shape:
        raise ValueError("delta shape %r does not match %r" % (delta.shape, base.shape))

    checked = check_tensor(tensor, base + delta, base, refuse_rescale=refuse_rescale)
    report = checkpoint.write_with({tensor: checked - base}, out_dir)
    rel = float(np.linalg.norm(delta) / np.linalg.norm(base)) if np.linalg.norm(base) else None
    return {
        "tensor": tensor,
        "relative_change": rel,
        "delta_sha256": hashlib.sha256(np.ascontiguousarray(delta).tobytes()).hexdigest(),
        "output": str(Path(out_dir)),
        "files": report["files"],
    }


def gate_metrics(metrics: Dict[str, float]) -> Dict[str, Any]:
    """Apply the default acceptance gates. Missing metrics count as failures."""
    verdict = evaluate(metrics)
    return {
        "passed": verdict.passed,
        "failures": verdict.failures,
        "missing": verdict.missing,
        "n_gates": verdict.n_gates,
        "summary": verdict.summary(),
    }


def wsp_audit(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Classify an edit as repair, acquisition, non-specific, or no-effect."""
    return assess_report(payload)


def save_wsp_evidence(
    out_dir: str,
    report: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Write a content-addressed WSP evidence bundle and return its address."""
    document = write_evidence(out_dir, report, metadata)
    return {
        "path": str(Path(out_dir)),
        "format": document["format"],
        "format_version": document["format_version"],
        "evidence_sha256": document["evidence_sha256"],
        "label": document["report"].get("label"),
        "accepted": document["report"].get("accepted"),
    }


def verify_wsp_binding(edit_dir: str, evidence_dir: str) -> Dict[str, Any]:
    """Verify an OpenWSP edit and WSP evidence bundle are bound together."""
    return verify_binding(edit_dir, evidence_dir).to_dict()


def plan_surgery(
    failure_type: str,
    layer_scores: Any = None,
    patch_report: Optional[Dict[str, Any]] = None,
    module_suffix: str = "mlp.down_proj",
    max_rel_norm: float = 0.02,
    min_score: float = 0.0,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Turn failure and site evidence into an explicit edit/diagnose route."""
    return _plan_surgery(
        failure_type,
        layer_scores=layer_scores,
        patch_report=patch_report,
        module_suffix=module_suffix,
        max_rel_norm=max_rel_norm,
        min_score=min_score,
        metadata=metadata,
    ).to_dict()


def diagnose_failure(
    base: Dict[str, Any],
    headroom: Optional[Dict[str, Any]] = None,
    mechanism: Optional[Dict[str, Any]] = None,
    target_metric: str = "accuracy",
    min_target_gap: float = 0.05,
    min_mechanism_gap: float = 0.05,
) -> Dict[str, Any]:
    """Classify obstruction vs capability gap from target and mechanism evidence."""
    from .planning import DiagnosisProtocol

    protocol = DiagnosisProtocol(
        target_metric=target_metric,
        min_target_gap=min_target_gap,
        min_mechanism_gap=min_mechanism_gap,
    )
    return _diagnose_failure(
        base, headroom=headroom, mechanism=mechanism, protocol=protocol
    ).to_dict()


def parse_intent(text: str) -> Dict[str, Any]:
    """Compile a plain-language teaching request into an exact contract."""
    intent = _parse_intent(text)
    payload = intent.to_dict()
    payload["command"] = intent.to_command()
    return payload
