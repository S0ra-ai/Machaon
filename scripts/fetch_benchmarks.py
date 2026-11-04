"""Download the benchmark files and record their provenance.

    HF_ENDPOINT=https://hf-mirror.com python scripts/fetch_benchmarks.py

Files land in ``$MACHAON_BENCH`` (default ``~/.cache/machaon/bench``), each with
a ``.provenance.json`` carrying source, revision, licence, item count and the
SHA-256 of the exact bytes that were read.
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huggingface_hub import hf_hub_download

from machaon.bench import Provenance, file_sha256, write_provenance

SOURCES = {
    "zsre-dev.jsonl": dict(
        repo="wangzn2001/zsre",
        filename="structured_zeroshot-dev-new_annotated_final.jsonl",
        name="zsRE (development)",
        revision="main",
        license="not declared by the mirror; upstream zsRE, Levy et al. 2017",
        notes="10,812 items with human-written rephrasings",
    ),
    "wikitext2-test.parquet": dict(
        repo="Salesforce/wikitext",
        filename="wikitext-2-raw-v1/test-00000-of-00001.parquet",
        name="WikiText-2 raw (test)",
        revision="main",
        license="CC-BY-SA-3.0 AND GFDL",
        notes="held-out perplexity corpus",
    ),
    "mmlu-test.parquet": dict(
        repo="cais/mmlu",
        filename="all/test-00000-of-00001.parquet",
        name="MMLU (test)",
        revision="main",
        license="MIT",
        notes="14,042 four-way questions across 57 subjects",
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.environ.get(
        "MACHAON_BENCH", str(Path.home() / ".cache" / "machaon" / "bench")))
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    retrieved = datetime.date.today().isoformat()

    for filename, spec in SOURCES.items():
        if args.only and filename not in args.only:
            continue
        target = out / filename
        if not target.exists():
            print("fetching %s ..." % filename)
            cached = hf_hub_download(
                repo_id=spec["repo"], filename=spec["filename"],
                repo_type="dataset", local_dir=str(out / "_hub"),
            )
            target.write_bytes(Path(cached).read_bytes())
        count = 0
        if filename.endswith(".jsonl"):
            with open(target, encoding="utf-8") as handle:
                count = sum(1 for line in handle if line.strip())
        else:
            import pyarrow.parquet as pq
            count = pq.read_metadata(target).num_rows
        write_provenance(target, Provenance(
            name=spec["name"], source="hf://datasets/%s/%s" % (spec["repo"], spec["filename"]),
            revision=spec["revision"], license=spec["license"],
            sha256=file_sha256(target), n_items=count, retrieved=retrieved,
            notes=spec["notes"],
        ))
        print("  %-24s %6d items  sha256 %s" % (filename, count, file_sha256(target)[:12]))
    print("done ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
