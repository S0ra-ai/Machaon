"""Generate one measured WSP benchmark reference case.

This is a deterministic threshold-classifier reference, not an LLM result.  It
exists to exercise the measured-provenance and benchmark gates end to end.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "examples" / "wsp_measured_smoke.json"


def _sha(array: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(array, dtype="<f8").tobytes()
    ).hexdigest()


def main() -> int:
    rng = np.random.default_rng(23)
    x = np.column_stack([
        rng.uniform(-2.0, 2.0, 400),
        rng.uniform(-1.0, 1.0, 400),
    ])
    y = (x[:, 0] >= 0.0).astype(np.int64)
    weights = np.array([1.0, 0.0], dtype=np.float64)
    base_bias = 0.8
    candidate_bias = 0.0
    control_bias = 1.6
    headroom_bias = 0.0

    def score(bias: float) -> float:
        return float(np.mean((x @ weights + bias >= 0.0) == y))

    base = score(base_bias)
    candidate = score(candidate_bias)
    control = score(control_bias)
    headroom = score(headroom_bias)
    parameters = np.concatenate([weights, [base_bias]])
    payload = {
        "format": "machaon.wsp.benchmark",
        "format_version": 1,
        "cases": [{
            "case_id": "reference-threshold-obstruction",
            "failure_type": "obstruction",
            "expected_repairable": True,
            "base": {"accuracy": base},
            "candidate": {"accuracy": candidate},
            "controls": {
                "opposite_sign_same_norm": {"accuracy": control},
            },
            "headroom": {
                "name": "oracle_threshold",
                "accuracy": headroom,
            },
            "gates": {"accuracy": [">=", 0.9]},
            "protocol": {
                "target_metric": "accuracy",
                "min_gain": 0.05,
                "control_margin": 0.05,
                "headroom_margin": 0.05,
            },
            "source_kind": "measured",
            "provenance": {
                "model": "toy-threshold-classifier-v1",
                "checkpoint_sha256": _sha(parameters),
                "dataset": "deterministic-threshold-reference-v1",
                "dataset_sha256": _sha(np.column_stack([x, y])),
                "evaluated_at": "2026-11-11T00:00:00Z",
                "seed": 23,
                "evaluation_script": "scripts/make_wsp_reference_case.py",
                "notes": "Pipeline reference only; not evidence about LLM WSP.",
            },
            "metadata": {
                "scope": "pipeline reference",
                "model_family": "threshold-classifier",
            },
        }],
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("wrote", OUT)
    print("base/candidate/control/headroom:",
          "%.3f %.3f %.3f %.3f" % (base, candidate, control, headroom))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
