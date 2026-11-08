"""Command line entry point.

Every command is a thin wrapper over the library.  Exit codes are part of the
contract, so the CLI can be used as a CI step without parsing text:

    0  success / gate passed
    1  gate rejected the edit, or validation failed
    2  malformed input, missing file, or unsupported checkpoint
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from importlib import metadata
from typing import Any, Dict

from . import __version__


def _error(message: str) -> int:
    print("error: %s" % message, file=sys.stderr)
    return 2


def _cmd_demo(_args) -> int:
    from .demo import main as demo_main
    demo_main()
    return 0


def _cmd_validate(args) -> int:
    from .schema import EditSpec
    try:
        with open(args.path) as f:
            spec = EditSpec.from_dict(json.load(f)).validate()
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return _error("%s: %s" % (args.path, exc))
    print("OK  %s -> %s (budget %.3f)"
          % (spec.op, spec.target.module, spec.constraints.max_rel_norm))
    return 0


def _cmd_gate(args) -> int:
    from .hygieia.gates import evaluate
    try:
        with open(args.path) as f:
            metrics = json.load(f)
    except (OSError, ValueError) as exc:
        return _error("%s: %s" % (args.path, exc))
    if not isinstance(metrics, dict):
        return _error("%s: metrics must be a JSON object" % args.path)
    verdict = evaluate(metrics)
    print(verdict.summary())
    return 0 if verdict.passed else 1


def _cmd_inspect(args) -> int:
    from .adapters import Checkpoint

    try:
        ckpt = Checkpoint(args.path)
    except (OSError, ValueError, ImportError) as exc:
        return _error(str(exc))
    names = ckpt.find(contains=args.contains, suffix=args.suffix, layer=args.layer)
    if args.limit is not None:
        names = names[: args.limit]
    rows = []
    for name in names:
        info = ckpt.info(name)
        rows.append({"name": name, "shape": list(info.shape), "dtype": info.dtype})
    if args.json:
        print(json.dumps({"path": str(ckpt.path), "tensors": rows}, indent=2))
        return 0
    for row in rows:
        print("%-64s %-20s %s" % (row["name"], str(row["shape"]), row["dtype"]))
    print("\n%d tensor(s)" % len(rows))
    return 0


def _optional_version(module: str) -> str:
    try:
        return metadata.version(module)
    except metadata.PackageNotFoundError:
        return "absent"


def _cmd_doctor(args) -> int:
    import numpy as np

    report: Dict[str, Any] = {
        "machaon": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "packages": {
            name: _optional_version(name)
            for name in (
                "torch",
                "transformers",
                "safetensors",
                "pyarrow",
                "mcp",
            )
        },
        "core": {
            "linalg": True,
            "contract_synthesis": True,
            "transactions": True,
            "delta_store": True,
            "gates": True,
            "openwsp_v1": True,
        },
    }
    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print("machaon %s" % report["machaon"])
    print("python  %s (%s)" % (report["python"], report["platform"]))
    print("numpy   %s" % report["numpy"])
    print("\ncore")
    for name, ready in report["core"].items():
        print("  %-14s %s" % (name, "ready" if ready else "unavailable"))
    print("\noptional")
    for name, version in report["packages"].items():
        print("  %-14s %s" % (name, version))
    if report["packages"]["torch"] == "absent":
        print("\ninstall real-model support with: pip install \"machaon[hf]\"")
    if report["packages"]["mcp"] == "absent":
        print("install the MCP server with:      pip install \"machaon[mcp]\"")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="machaon",
        description="Direct weight surgery for LLMs. Extract the arrow, leave the body intact.",
    )
    parser.add_argument("--version", action="version", version="machaon " + __version__)
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("demo", help="run the synthetic end-to-end demo (no downloads)")
    p_val = sub.add_parser("validate", help="validate an edit spec JSON")
    p_val.add_argument("path")
    p_gate = sub.add_parser("gate", help="check metrics JSON against the acceptance gates")
    p_gate.add_argument("path")
    p_ins = sub.add_parser("inspect", help="list tensors in a safetensors checkpoint")
    p_ins.add_argument("path")
    p_ins.add_argument("--contains", default=None)
    p_ins.add_argument("--suffix", default=None)
    p_ins.add_argument("--layer", type=int, default=None)
    p_ins.add_argument("--limit", type=int, default=None)
    p_ins.add_argument("--json", action="store_true", help="machine-readable output")
    p_doc = sub.add_parser("doctor", help="report installed capabilities and versions")
    p_doc.add_argument("--json", action="store_true", help="machine-readable output")

    args = parser.parse_args(argv)
    if args.command == "demo":
        return _cmd_demo(args)
    if args.command == "validate":
        return _cmd_validate(args)
    if args.command == "gate":
        return _cmd_gate(args)
    if args.command == "inspect":
        return _cmd_inspect(args)
    if args.command == "doctor":
        return _cmd_doctor(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
