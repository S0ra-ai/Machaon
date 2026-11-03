"""Command line entry point."""
from __future__ import annotations

import argparse
import json
import sys

from . import __version__


def _cmd_demo(_args) -> int:
    from .demo import main as demo_main
    demo_main()
    return 0


def _cmd_validate(args) -> int:
    from .schema import EditSpec
    with open(args.path) as f:
        spec = EditSpec.from_dict(json.load(f)).validate()
    print("OK  %s -> %s (budget %.3f)"
          % (spec.op, spec.target.module, spec.constraints.max_rel_norm))
    return 0


def _cmd_gate(args) -> int:
    from .hygieia.gates import evaluate
    with open(args.path) as f:
        metrics = json.load(f)
    verdict = evaluate(metrics)
    print(verdict.summary())
    return 0 if verdict.passed else 1


def _cmd_inspect(args) -> int:
    from .adapters import Checkpoint

    ckpt = Checkpoint(args.path)
    names = ckpt.find(contains=args.contains, suffix=args.suffix, layer=args.layer)
    if args.limit is not None:
        names = names[: args.limit]
    for name in names:
        info = ckpt.info(name)
        print("%-64s %-20s %s" % (name, str(list(info.shape)), info.dtype))
    print("\n%d tensor(s)" % len(names))
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

    args = parser.parse_args(argv)
    if args.command == "demo":
        return _cmd_demo(args)
    if args.command == "validate":
        return _cmd_validate(args)
    if args.command == "gate":
        return _cmd_gate(args)
    if args.command == "inspect":
        return _cmd_inspect(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
