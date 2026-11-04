"""Apply a delta to a checkpoint on disk, without loading the whole model.

    python examples/file_edit.py --model /tmp/qwen15b --layer 20 --out /tmp/edited

This is the path that scales: one tensor in, one tensor out, everything else
copied bit for bit. The edit here is mechanical (a synthetic key/value pair) —
it exercises the I/O, not the science.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from machaon.adapters import Checkpoint
from machaon.linalg import rank1_constrained


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--suffix", default="mlp.down_proj.weight")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--single-shard", action="store_true",
                        help="rewrite only the shard holding the tensor; hardlink the rest")
    parser.add_argument("--budget", type=float, default=0.005,
                        help="target relative change of the delta")
    args = parser.parse_args()

    started = time.time()
    ckpt = Checkpoint(args.model)
    summary = ckpt.summary()
    print("source   : %d tensors, %.1fM params, %d shard(s)" % (
        summary["tensors"], summary["parameters"] / 1e6, summary["shards"]))

    matches = ckpt.find(layer=args.layer, suffix=args.suffix)
    if not matches:
        print("no tensor matches layer %d / %s" % (args.layer, args.suffix))
        return 1
    name = matches[0]
    print("target   : %s %s" % (name, list(ckpt.info(name).shape)))

    W = ckpt.read(name)
    rng = np.random.default_rng(args.seed)
    key = rng.normal(size=W.shape[1])
    unit = rng.normal(size=W.shape[0])
    unit /= np.linalg.norm(unit)
    # rank1_constrained gives ‖delta‖ = ‖value - W key‖ / ‖key‖, so size the
    # shift to spend a realistic fraction of the tensor's norm.
    shift = args.budget * np.linalg.norm(W) * np.linalg.norm(key) * unit
    with np.errstate(all="ignore"):  # macOS Accelerate sets spurious FP flags
        value = W @ key + shift
    _, delta = rank1_constrained(W, key, value)
    print("delta    : rel %.4f" % (np.linalg.norm(delta) / np.linalg.norm(W)))

    out = Path(args.out)
    if args.single_shard:
        report = ckpt.write_tensor(name, W + delta, out)
        print("shard    : %s (linked %d, copied %d)" % (
            report["shard"], len(report["linked"]), len(report["copied"])))
    else:
        ckpt.write_with({name: delta}, out)
    print("wrote    : %s (%.1fs)" % (out, time.time() - started))

    reloaded = Checkpoint(out)
    changed = reloaded.read(name).astype(np.float64) - W
    print("reload   : %d tensors, max |Δ| %.3e" % (
        reloaded.summary()["tensors"], float(np.abs(changed).max())))

    untouched = next(n for n in ckpt.names() if n != name)
    same = np.array_equal(ckpt.read(untouched, dtype=None), reloaded.read(untouched, dtype=None))
    print("untouched: %s identical=%s" % (untouched, same))
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
