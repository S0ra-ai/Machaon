"""WikiText-2 raw, used for held-out perplexity."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional


def load_documents(path: str | Path, min_chars: int = 400,
                   limit: Optional[int] = None) -> List[str]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise ImportError("load_wikitext needs pyarrow: pip install pyarrow") from exc

    table = pq.read_table(path, columns=["text"])
    documents = []
    for value in table.column("text").to_pylist():
        text = (value or "").strip()
        if len(text) >= min_chars:
            documents.append(text)
        if limit is not None and len(documents) >= limit:
            break
    return documents
