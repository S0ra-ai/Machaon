"""Compatibility wrapper for :mod:`machaon.fetch`.

    python scripts/fetch_model.py --repo Qwen/Qwen2.5-1.5B --dest ./qwen15b

The downloader lives in the package so the CLI and the Web console use the
same endpoint fallback and header validation.  This wrapper keeps the old
script path, its imports and its test hooks working from a checkout.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from machaon.fetch import (  # noqa: E402,F401
    ENDPOINTS,
    WANTED,
    check_safetensors,
    download,
    download_with_fallback,
    fetch_model,
    file_url,
    list_files,
    main,
)

if __name__ == "__main__":
    sys.exit(main())
