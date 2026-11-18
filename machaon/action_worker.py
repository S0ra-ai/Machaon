"""Isolated worker for :mod:`machaon.action`.

The parent validates the source before launch and again in this process.  The
worker then executes only the validated AST with empty builtins and a small
allowlist of pure array functions.  It is a bounded computation process, not a
general-purpose Python security boundary.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .action import SAFE_FUNCTIONS, validate_action_source

__all__ = ["main"]


def _limit_resources(memory_mb: int, timeout_seconds: float) -> None:
    if os.name != "posix":
        return
    try:
        import resource

        memory = int(memory_mb) * 1024 * 1024
        cpu = max(1, int(math.ceil(float(timeout_seconds))) + 1)
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
    except (ImportError, OSError, ValueError):
        # The wall-clock deadline remains the hard bound on platforms where
        # address-space limits are unavailable.
        pass


def _load_array(archive: Any, name: str):
    if name not in archive:
        return None
    return np.array(archive[name], dtype=np.float64, copy=True)


def _write_response(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="run one constrained Machaon action")
    parser.add_argument("--request", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--response", required=True)
    args = parser.parse_args(argv)

    response_path = Path(args.response)
    try:
        request = json.loads(Path(args.request).read_text(encoding="utf-8"))
        _limit_resources(int(request["memory_mb"]), float(request["timeout_seconds"]))
        tree = validate_action_source(str(request["source"]))
        with np.load(args.input, allow_pickle=False) as archive:
            W = _load_array(archive, "W")
            keys = _load_array(archive, "keys")
            values = _load_array(archive, "values")
            preserve = _load_array(archive, "preserve")
        if W is None or W.ndim != 2:
            raise ValueError("worker input needs a 2-D W array")
        if not np.isfinite(W).all():
            raise ValueError("worker input W is not finite")

        started = time.monotonic()
        environment: Dict[str, Any] = {
            "__builtins__": {},
            "W": W,
            "keys": keys,
            "values": values,
            "preserve": preserve,
            "max_rel_norm": float(request["max_rel_norm"]),
        }
        environment.update(SAFE_FUNCTIONS)
        code = compile(tree, "<machaon-action>", "exec")
        exec(code, environment)
        function = environment.get("edit")
        if not callable(function):
            raise ValueError("action did not define edit(W)")
        delta = function(W)
        delta = np.asarray(delta, dtype=np.float64)
        if delta.shape != W.shape:
            raise ValueError(
                "action returned shape %r, expected %r" % (delta.shape, W.shape)
            )
        if not np.isfinite(delta).all():
            raise ValueError("action returned non-finite values")
        np.save(args.output, delta, allow_pickle=False)
        _write_response(response_path, {
            "ok": True,
            "elapsed_ms": (time.monotonic() - started) * 1000.0,
        })
        return 0
    except BaseException as exc:  # the response must survive MemoryError too
        try:
            _write_response(response_path, {
                "ok": False,
                "error": "%s: %s" % (type(exc).__name__, str(exc)[:1000]),
            })
        except OSError:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
