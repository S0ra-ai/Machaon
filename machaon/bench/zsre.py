"""zsRE, the standard zero-shot relation extraction editing benchmark.

10,812 development items from Levy et al. (2017), as distributed for the MEND
and SERAC evaluations. Each item carries the question, the gold answer, and a
set of human-written rephrasings — which is what makes it useful: the
rephrasings are a held-out generalisation set, not something we wrote.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class EditCase:
    case_id: str
    question: str
    answer: str
    rephrases: Sequence[str]

    @property
    def prompts(self) -> List[str]:
        return [self.question, *self.rephrases]


def load(path: str | Path, limit: Optional[int] = None,
         require_rephrases: bool = True) -> List[EditCase]:
    cases: List[EditCase] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            outputs = record.get("output") or []
            if not outputs:
                continue
            answer = outputs[0].get("answer")
            rephrases = tuple(record.get("rephrases") or ())
            if not answer:
                continue
            if require_rephrases and not rephrases:
                continue
            cases.append(EditCase(
                case_id=str(record.get("id", len(cases))),
                question=record["input"],
                answer=answer,
                rephrases=rephrases,
            ))
            if limit is not None and len(cases) >= limit:
                break
    return cases
