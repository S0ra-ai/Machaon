"""Download a model, preferring whatever endpoint actually works here.

    python scripts/fetch_model.py --repo Qwen/Qwen2.5-1.5B --dest ./qwen15b

Tries the Hugging Face endpoint, then the configured mirror, then ModelScope.
ModelScope is often several times faster from mainland China. Every downloaded
``.safetensors`` file is checked by reading its header, not by loading weights.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ENDPOINTS = [
    ("huggingface", "https://huggingface.co"),
    ("hf-mirror", "https://hf-mirror.com"),
    ("modelscope", "https://www.modelscope.cn/api/v1/models"),
]

WANTED = (".safetensors", ".json", ".txt", "tokenizer", "vocab", "merges")


def _get(url: str, timeout: int = 30) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "machaon-fetch"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def list_files(repo: str, endpoint: str) -> list:
    if endpoint.endswith("/api/v1/models"):
        payload = json.loads(_get("%s/%s/repo/files?Revision=master" % (endpoint, repo)))
        return [item["Path"] for item in payload.get("Data", {}).get("Files", [])]
    payload = json.loads(_get("%s/api/models/%s" % (endpoint, repo)))
    return [item["rfilename"] for item in payload.get("siblings", [])]


def file_url(repo: str, name: str, endpoint: str) -> str:
    if endpoint.endswith("/api/v1/models"):
        return "%s/%s/repo?Revision=master&FilePath=%s" % (
            endpoint, repo, urllib.parse.quote(name))
    return "%s/%s/resolve/main/%s" % (endpoint, repo, urllib.parse.quote(name))


def download(repo: str, name: str, target: Path, endpoint: str) -> None:
    url = file_url(repo, name, endpoint)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    start = time.time()
    try:
        with urllib.request.urlopen(url, timeout=120) as response, open(temporary, "wb") as handle:
            expected = response.headers.get("Content-Length")
            written = 0
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
                written += len(chunk)
        if expected is not None and written != int(expected):
            raise IOError(
                "incomplete download for %s: received %d of %s bytes"
                % (name, written, expected)
            )
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    temporary.replace(target)
    elapsed = max(time.time() - start, 1e-6)
    print("  %-40s %8.1f MB  %6.2f MB/s" % (
        name, target.stat().st_size / 1e6, target.stat().st_size / 1e6 / elapsed))


def download_with_fallback(
    repo: str,
    name: str,
    target: Path,
    candidates: list,
    downloader=download,
    validator=None,
) -> tuple:
    """Try the endpoint that listed the repo, then the remaining endpoints."""
    errors = []
    for label, endpoint in candidates:
        try:
            downloader(repo, name, target, endpoint)
            if validator is not None:
                try:
                    validator(target)
                except Exception:
                    target.unlink(missing_ok=True)
                    raise
            return label, endpoint
        except Exception as exc:
            errors.append("%s: %s" % (label, exc))
            print("  %-40s fallback from %s: %s" % (name, label, exc))
    raise IOError(
        "could not download %s from any endpoint: %s"
        % (name, "; ".join(errors))
    )


def check_safetensors(path: Path) -> dict:
    with open(path, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        if length <= 0 or length > 100 * 1024 * 1024:
            raise ValueError("implausible header length %d in %s" % (length, path))
        header = json.loads(handle.read(length))
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    total = sum(int(math.prod(v["shape"])) for v in tensors.values())
    return {"tensors": len(tensors), "parameters": total}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--endpoint", default=None,
                        help="force one of: huggingface, hf-mirror, modelscope")
    args = parser.parse_args()

    dest = Path(args.dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)

    candidates = [e for e in ENDPOINTS if args.endpoint in (None, e[0])]
    if args.endpoint:
        candidates = [e for e in ENDPOINTS if e[0] == args.endpoint]
        if not candidates:
            parser.error("unknown endpoint")

    chosen = None
    files = []
    for label, endpoint in candidates:
        try:
            files = list_files(args.repo, endpoint)
            chosen = (label, endpoint)
            print("endpoint %s: %d files" % (label, len(files)))
            break
        except Exception as exc:
            print("endpoint %s unavailable: %s" % (label, exc))
    if chosen is None:
        return 1

    wanted = [f for f in files if any(w in f for w in WANTED) and not f.endswith(".gitattributes")]
    for name in wanted:
        target = dest / name
        if target.exists() and target.stat().st_size > 0:
            print("  %-40s cached" % name)
            continue
        validator = check_safetensors if name.endswith(".safetensors") else None
        download_with_fallback(
            args.repo, name, target, candidates, validator=validator)

    print("\nchecking safetensors headers")
    for path in sorted(dest.glob("*.safetensors")):
        info = check_safetensors(path)
        print("  %-40s %d tensors, %.1fM params" % (
            path.name, info["tensors"], info["parameters"] / 1e6))
    print("\ndone ->", dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
