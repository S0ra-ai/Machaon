"""Fail fast when a checkpoint cannot fit.

Thrashing is worse than refusing: the process appears alive, makes no progress,
and takes the machine down with it. This module estimates weight bytes from the
safetensors header and compares them with what the OS says is available.
"""
from __future__ import annotations

import re
import subprocess
import sys
from typing import Any, Dict, Optional

__all__ = ["available_bytes", "weights_bytes", "check_fit"]

DTYPE_BYTES = {
    "BF16": 2, "F16": 2, "F32": 4, "F64": 8,
    "I8": 1, "U8": 1, "I16": 2, "I32": 4, "I64": 8, "BOOL": 1,
}


def available_bytes() -> Optional[int]:
    """Bytes the OS says it can hand out now, or None when unknown."""
    try:
        if sys.platform == "darwin":
            out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
            page = int(re.search(r"page size of (\d+) bytes", out).group(1))
            free = int(re.search(r"Pages free:\s+(\d+)", out).group(1))
            inactive = int(re.search(r"Pages inactive:\s+(\d+)", out).group(1))
            return (free + inactive) * page
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, AttributeError, subprocess.SubprocessError):
        return None
    return None


def weights_bytes(summary: Dict[str, Any]) -> int:
    """Size of the weights themselves, from ``Checkpoint.summary()``."""
    total = 0
    for dtype, count in summary.get("dtypes", {}).items():
        if dtype not in DTYPE_BYTES:
            raise ValueError("unknown dtype in summary: %s" % dtype)
        total += int(count) * DTYPE_BYTES[dtype]
    return total


def check_fit(summary: Dict[str, Any], fraction: float = 0.6) -> Dict[str, Any]:
    """Would this checkpoint fit in ``fraction`` of currently available memory?

    ``fraction`` leaves room for activations, the tokenizer and the rest of the
    program. The default of 0.6 comes from measurement: a 7.6B checkpoint at 67%
    of reported-available memory (22.7 GB) thrashed until the run was stopped.
    """
    needed = weights_bytes(summary)
    available = available_bytes()
    if available is None:
        return {
            "ok": True, "unknown": True, "weights_bytes": needed,
            "available_bytes": None, "ratio": None,
            "reason": "available memory could not be determined",
        }
    usable = int(available * fraction)
    ok = needed <= usable
    return {
        "ok": ok,
        "unknown": False,
        "weights_bytes": needed,
        "available_bytes": available,
        "usable_bytes": usable,
        "ratio": needed / available if available else None,
        "reason": (
            "weights %.1f GB fit in %.1f GB available" % (needed / 1e9, available / 1e9)
            if ok else
            "weights %.1f GB exceed the %.1f GB budget (%.1f GB available)"
            % (needed / 1e9, usable / 1e9, available / 1e9)
        ),
    }
