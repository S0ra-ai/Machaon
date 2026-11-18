"""Fetch a safetensors checkpoint without loading it.

The core package stays NumPy-only: this module uses the standard library to
list and download files, tries the Hugging Face endpoint, a mirror, and
ModelScope, and validates every ``.safetensors`` header before accepting it.
It never imports torch and never materialises a weight tensor.

    machaon fetch --starter qwen05b --dest ./models/Qwen2.5-0.5B
    machaon fetch --repo Qwen/Qwen2.5-0.5B --dest ./models/qwen
    machaon fetch --list
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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "ENDPOINTS",
    "RECOMMENDED",
    "WANTED",
    "check_safetensors",
    "download",
    "download_with_fallback",
    "fetch_model",
    "file_url",
    "list_files",
    "main",
]

ENDPOINTS = [
    ("huggingface", "https://huggingface.co"),
    ("hf-mirror", "https://hf-mirror.com"),
    ("modelscope", "https://www.modelscope.cn/api/v1/models"),
]

WANTED = (".safetensors", "tokenizer", "sentencepiece", "chat_template",
          "config.json", "vocab.json", "merges.txt")

_EXACT_FILES = {
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "preprocessor_config.json",
    "processor_config.json",
    "quantize_config.json",
    "vocab.json",
    "merges.txt",
}


def _wanted_file(name: str) -> bool:
    """Keep weights and the small files a tokenizer/config needs.

    Training/eval result files are deliberately excluded: a checkpoint
    download should not pull experiment logs just because they are JSON.
    """
    base = name.rsplit("/", 1)[-1]
    if base.endswith(".safetensors"):
        return True
    if base in _EXACT_FILES:
        return True
    return base.startswith(("tokenizer", "sentencepiece", "chat_template"))

RECOMMENDED: Dict[str, Dict[str, str]] = {
    "smol135m": {
        "repo": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "name": "SmolLM2-135M",
        "size": "~270 MB",
        "note": "smallest useful chat checkpoint; good for a first run",
    },
    "qwen05b": {
        "repo": "Qwen/Qwen2.5-0.5B",
        "name": "Qwen2.5-0.5B",
        "size": "~1 GB",
        "note": "the checkpoint used by the README examples",
    },
    "gpt2": {
        "repo": "gpt2",
        "name": "gpt2",
        "size": "~500 MB",
        "note": "classic baseline; Conv1D orientation path is tested",
    },
}


def _emit(progress: Optional[Callable[[Dict[str, Any]], None]], **data: Any) -> None:
    if progress is None:
        return
    try:
        progress(data)
    except Exception:
        # A UI callback must never abort a download.
        pass


def _get(url: str, timeout: int = 30) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "machaon-fetch"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def list_files(repo: str, endpoint: str) -> List[str]:
    if endpoint.endswith("/api/v1/models"):
        payload = json.loads(
            _get("%s/%s/repo/files?Revision=master" % (endpoint, repo))
        )
        return [item["Path"] for item in payload.get("Data", {}).get("Files", [])]
    payload = json.loads(_get("%s/api/models/%s" % (endpoint, repo)))
    return [item["rfilename"] for item in payload.get("siblings", [])]


def file_url(repo: str, name: str, endpoint: str) -> str:
    if endpoint.endswith("/api/v1/models"):
        return "%s/%s/repo?Revision=master&FilePath=%s" % (
            endpoint, repo, urllib.parse.quote(name)
        )
    return "%s/%s/resolve/main/%s" % (
        endpoint, repo, urllib.parse.quote(name)
    )


def download(
    repo: str,
    name: str,
    target: Path,
    endpoint: str,
    progress: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> None:
    url = file_url(repo, name, endpoint)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    start = time.time()
    last_report = 0.0
    try:
        with urllib.request.urlopen(url, timeout=120) as response, open(
            temporary, "wb"
        ) as handle:
            expected = response.headers.get("Content-Length")
            total = int(expected) if expected is not None else None
            written = 0
            _emit(
                progress,
                phase="download",
                file=name,
                bytes=0,
                total=total,
            )
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
                written += len(chunk)
                now = time.time()
                if now - last_report >= 0.5 or (
                    total is not None and written >= total
                ):
                    last_report = now
                    _emit(
                        progress,
                        phase="download",
                        file=name,
                        bytes=written,
                        total=total,
                    )
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
    print(
        "  %-40s %8.1f MB  %6.2f MB/s"
        % (name, target.stat().st_size / 1e6, target.stat().st_size / 1e6 / elapsed)
    )


def download_with_fallback(
    repo: str,
    name: str,
    target: Path,
    candidates: List[Tuple[str, str]],
    downloader: Optional[Callable[[str, str, Path, str], None]] = None,
    validator: Optional[Callable[[Path], Any]] = None,
    progress: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[str, str]:
    """Try the endpoint that listed the repo, then the remaining endpoints."""
    if downloader is None:
        def download_one(repo_, name_, target_, endpoint_):
            try:
                download(repo_, name_, target_, endpoint_, progress=progress)
            except TypeError as exc:
                if "progress" not in str(exc):
                    raise
                # Backward compatibility for a 4-argument custom downloader.
                download(repo_, name_, target_, endpoint_)

        downloader = download_one

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


def check_safetensors(path: Path) -> Dict[str, Any]:
    with open(path, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        if length <= 0 or length > 100 * 1024 * 1024:
            raise ValueError("implausible header length %d in %s" % (length, path))
        header = json.loads(handle.read(length))
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    total = sum(int(math.prod(v["shape"])) for v in tensors.values())
    return {"tensors": len(tensors), "parameters": total}


def fetch_model(
    repo: str,
    dest: str | Path,
    *,
    endpoint: Optional[str] = None,
    require_new: bool = False,
    progress: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Download a safetensors checkpoint into ``dest``.

    Existing files are reused, so an interrupted download can be resumed.
    ``require_new=True`` refuses a non-empty destination; the Web console uses
    it to avoid mixing two models in one output directory.
    """
    if not isinstance(repo, str) or not repo.strip():
        raise ValueError("repo must be a non-empty string")
    destination = Path(dest).expanduser()
    if require_new and destination.exists():
        try:
            non_empty = any(destination.iterdir())
        except OSError:
            non_empty = True
        if non_empty:
            raise FileExistsError("destination already exists: %s" % destination)
    destination.mkdir(parents=True, exist_ok=True)

    candidates = list(ENDPOINTS)
    if endpoint is not None:
        candidates = [item for item in ENDPOINTS if item[0] == endpoint]
        if not candidates:
            raise ValueError(
                "endpoint must be one of %s" % ", ".join(name for name, _ in ENDPOINTS)
            )

    chosen: Optional[Tuple[str, str]] = None
    files: List[str] = []
    errors = []
    for label, base in candidates:
        _emit(progress, phase="list", repo=repo, endpoint=label)
        try:
            files = list_files(repo, base)
            chosen = (label, base)
            print("endpoint %s: %d files" % (label, len(files)))
            break
        except Exception as exc:
            errors.append("%s: %s" % (label, exc))
            print("endpoint %s unavailable: %s" % (label, exc))
    if chosen is None:
        raise IOError(
            "could not list %s from any endpoint: %s"
            % (repo, "; ".join(errors))
        )

    wanted = [
        name for name in files
        if _wanted_file(name) and not name.endswith(".gitattributes")
    ]
    weights = [name for name in wanted if name.endswith(".safetensors")]
    if not weights:
        raise ValueError(
            "%s has no .safetensors weights; choose a safetensors checkpoint"
            % repo
        )

    downloaded: List[str] = []
    cached: List[str] = []
    _emit(
        progress,
        phase="plan",
        repo=repo,
        files=len(wanted),
        cached=sum(
            1 for name in wanted
            if (destination / name).exists()
            and (destination / name).stat().st_size > 0
        ),
    )
    for index, name in enumerate(wanted, 1):
        target = destination / name
        if target.exists() and target.stat().st_size > 0:
            cached.append(name)
            print("  %-40s cached" % name)
            _emit(
                progress,
                phase="file",
                file=name,
                index=index,
                count=len(wanted),
                cached=True,
            )
            continue
        _emit(
            progress,
            phase="file",
            file=name,
            index=index,
            count=len(wanted),
            cached=False,
        )
        validator = check_safetensors if name.endswith(".safetensors") else None
        download_with_fallback(
            repo, name, target, candidates,
            validator=validator, progress=progress,
        )
        downloaded.append(name)

    index_path = destination / "model.safetensors.index.json"
    if index_path.exists():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map", {})
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("model.safetensors.index.json has no weight_map")
        missing = sorted(
            name for name in set(weight_map.values())
            if not (destination / name).exists()
        )
        if missing:
            raise ValueError(
                "index references missing shards: %s" % ", ".join(missing)
            )
    infos = [check_safetensors(path) for path in sorted(destination.glob("*.safetensors"))]
    total_parameters = sum(info["parameters"] for info in infos)
    total_tensors = sum(info["tensors"] for info in infos)
    report = {
        "repo": repo,
        "dest": str(destination),
        "endpoint": chosen[0],
        "files": sorted(downloaded + cached),
        "downloaded": sorted(downloaded),
        "cached": sorted(cached),
        "tensors": total_tensors,
        "parameters": total_parameters,
        "shards": len(infos),
    }
    _emit(
        progress,
        phase="done",
        repo=repo,
        dest=report["dest"],
        parameters=total_parameters,
        shards=len(infos),
    )
    return report


def _print_recommended() -> None:
    print("recommended starter checkpoints")
    for key, item in RECOMMENDED.items():
        print(
            "  %-10s %-28s %-9s %s"
            % (key, item["repo"], item["size"], item["note"])
        )
    print("\nusage: machaon fetch --starter qwen05b --dest ./models/Qwen2.5-0.5B")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="machaon fetch",
        description="Download a safetensors checkpoint without loading it.",
    )
    parser.add_argument("--repo", default=None, help="Hugging Face repository id")
    parser.add_argument(
        "--starter", choices=tuple(RECOMMENDED), default=None,
        help="use a recommended starter checkpoint",
    )
    parser.add_argument("--dest", default=None, help="destination directory")
    parser.add_argument(
        "--endpoint", default=None,
        choices=tuple(name for name, _ in ENDPOINTS),
        help="force one endpoint instead of trying all",
    )
    parser.add_argument("--list", action="store_true", help="list starters and exit")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    if args.list:
        if args.json:
            print(json.dumps(RECOMMENDED, indent=2, sort_keys=True))
        else:
            _print_recommended()
        return 0
    if args.starter:
        item = RECOMMENDED[args.starter]
        repo = item["repo"]
        default_name = item["name"]
    elif args.repo:
        repo = args.repo
        default_name = repo.rsplit("/", 1)[-1]
    else:
        print("error: pass --repo, --starter or --list", file=sys.stderr)
        return 2
    dest = args.dest or str(Path("models") / default_name)
    try:
        report = fetch_model(repo, dest, endpoint=args.endpoint)
    except (OSError, ValueError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            "\ndone -> %s\n%s: %d shard(s), %d tensors, %.1fM parameters"
            % (
                report["dest"],
                report["endpoint"],
                report["shards"],
                report["tensors"],
                report["parameters"] / 1e6,
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
