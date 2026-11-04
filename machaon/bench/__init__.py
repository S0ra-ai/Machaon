"""Open benchmarks, with provenance attached.

Every loader returns items plus a :class:`Provenance` record naming the source,
the revision, the licence and the SHA-256 of the exact file that was read. A
number without that record is not a result, it is an anecdote.
"""
from __future__ import annotations

from .mmlu import MCQItem
from .mmlu import accuracy as mmlu_accuracy
from .mmlu import load as load_mmlu
from .provenance import Provenance, file_sha256, read_provenance, write_provenance
from .wikitext import load_documents as load_wikitext
from .zsre import EditCase
from .zsre import load as load_zsre

__all__ = [
    "Provenance",
    "file_sha256",
    "read_provenance",
    "write_provenance",
    "EditCase",
    "load_zsre",
    "MCQItem",
    "load_mmlu",
    "mmlu_accuracy",
    "load_wikitext",
]
