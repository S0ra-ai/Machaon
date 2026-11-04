"""Provenance for benchmark files."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Provenance:
    name: str
    source: str
    revision: str
    license: str
    sha256: str
    n_items: int
    retrieved: str
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "Provenance":
        return cls(**payload)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_provenance(data_path: str | Path, record: Provenance) -> Path:
    target = Path(str(data_path) + ".provenance.json")
    target.write_text(json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n")
    return target


def read_provenance(data_path: str | Path) -> Provenance:
    target = Path(str(data_path) + ".provenance.json")
    return Provenance.from_dict(json.loads(target.read_text()))


def verify(data_path: str | Path) -> Provenance:
    """Return the record, raising if the file no longer matches its digest."""
    record = read_provenance(data_path)
    actual = file_sha256(data_path)
    if actual != record.sha256:
        raise ValueError(
            "benchmark file changed: %s expected %s got %s"
            % (data_path, record.sha256[:12], actual[:12])
        )
    return record
