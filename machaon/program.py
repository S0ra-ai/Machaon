"""Portable, declarative weight-space programs.

A program is a small text file that names a checkpoint, one or more teaching
facts, and the safety budget for the whole run.  It is deliberately a thin
composition layer over :mod:`machaon.teach_many` and
:mod:`machaon.check`: the program does not introduce a second edit path.

The text format is intentionally line-oriented so it can be reviewed and
diffed without a parser library::

    version: 1
    model: ./models/SmolLM2-135M
    fetch: smol135m
    output: ./models/hello

    teach: The Eiffel Tower is in the city of -> Rome
    teach: The capital of France is -> Paris
    protect: The capital of Germany is

Relative model and output paths resolve from the program file, not from the
current working directory.  The same object can also be written as JSON.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .budget import EditBudget
from .intent import IntentError, parse_intent
from .recipes import TeachPolicy

__all__ = [
    "ProgramError",
    "ProgramReportError",
    "WeightProgram",
    "ProgramRun",
    "PROGRAM_REPORT_FORMAT",
    "PROGRAM_REPORT_VERSION",
    "PROGRAM_REPORT_FILENAME",
    "load_program",
    "parse_program",
    "program_report_sha256",
    "program_report_checkpoint",
    "parse_program_report",
    "read_program_report",
    "render_program_text",
    "verify_program_report",
    "write_program_report",
    "run_program",
]


class ProgramError(ValueError):
    """A program is syntactically valid JSON/text but not a valid contract."""


class ProgramReportError(ProgramError):
    """A program report is malformed, corrupted, or not self-consistent."""


PROGRAM_REPORT_FORMAT = "machaon.program.report/1"
PROGRAM_REPORT_VERSION = 1
PROGRAM_REPORT_FILENAME = "program-report.json"

_SINGLE_FIELDS = {
    "version",
    "name",
    "model",
    "fetch",
    "starter",
    "output",
    "budget",
    "max_rel_norm",
    "max_kl",
    "min_panel_agreement",
    "layers",
    "solver",
    "steps",
    "lr",
    "max_layers",
    "dtype",
    "device",
    "quantize",
    "verify",
    "continue_on_reject",
    "keep_steps",
}
_REPEAT_FIELDS = {"teach", "protect"}
_KNOWN_FIELDS = _SINGLE_FIELDS | _REPEAT_FIELDS
_DTYPES = {"auto", "float32", "float16", "bfloat16"}
_SOLVERS = {"direction", "gradient"}
_QUANTIZE = {"int4", "int8"}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _positive_float(value: Any, name: str, *, maximum: Optional[float] = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ProgramError("%s must be a number" % name) from exc
    if not math.isfinite(number) or number <= 0:
        raise ProgramError("%s must be finite and positive" % name)
    if maximum is not None and number > maximum:
        raise ProgramError("%s must be <= %s" % (name, maximum))
    return number


def _bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ProgramError("%s must be true or false" % name)


def _resolve_path(value: str, base_dir: Path) -> str:
    path = Path(value.strip()).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def parse_layers(value: Any) -> Optional[Tuple[int, ...]]:
    """Parse ``18,20,22`` / ``18-22`` or a JSON list into a layer tuple."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        parts = [str(item) for item in value]
    else:
        parts = str(value).replace(" ", "").split(",")
    layers: List[int] = []
    for part in parts:
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            try:
                first, last = int(start), int(end)
            except ValueError as exc:
                raise ProgramError("invalid layer range %r" % part) from exc
            if last < first:
                raise ProgramError("layer range %r is descending" % part)
            layers.extend(range(first, last + 1))
        else:
            try:
                layers.append(int(part))
            except ValueError as exc:
                raise ProgramError("invalid layer %r" % part) from exc
    if any(layer < 0 for layer in layers):
        raise ProgramError("layers must be non-negative")
    return tuple(sorted(set(layers))) if layers else None


def _normalize_fact(item: Any, index: int) -> Dict[str, Any]:
    if isinstance(item, str):
        sentence = item.strip()
        if not sentence:
            raise ProgramError("teach entry %d is empty" % index)
        try:
            intent = parse_intent(sentence)
        except IntentError as exc:
            raise ProgramError("teach entry %d: %s" % (index, exc)) from exc
        return {
            "say": sentence,
            "prompt": intent.prompt,
            "answer": intent.answer,
            "confidence": intent.confidence,
            "source": intent.source,
            "notes": list(intent.notes),
        }
    if isinstance(item, Mapping):
        if "say" in item:
            return _normalize_fact(str(item["say"]), index)
        prompt = item.get("prompt")
        answer = item.get("answer")
        if isinstance(prompt, str) and prompt.strip() and isinstance(answer, str):
            return {
                "say": None,
                "prompt": prompt,
                "answer": answer,
                "confidence": "high",
                "source": "arguments",
                "notes": [],
            }
        raise ProgramError(
            "teach entry %d needs 'say', or string 'prompt' and 'answer'" % index
        )
    raise ProgramError("teach entry %d must be a sentence or an object" % index)


def _normalize_protect(items: Any) -> Tuple[str, ...]:
    if items is None:
        return ()
    if isinstance(items, str):
        items = [items]
    if not isinstance(items, (list, tuple)):
        raise ProgramError("protect must be a list of prompts")
    prompts: List[str] = []
    for index, item in enumerate(items, 1):
        if not isinstance(item, str) or not item.strip():
            raise ProgramError("protect entry %d must be a non-empty prompt" % index)
        prompt = item.strip()
        if prompt not in prompts:
            prompts.append(prompt)
    return tuple(prompts)


def _parse_text(text: str, base_dir: Path, source: str) -> Dict[str, Any]:
    values: Dict[str, Any] = {"teach": [], "protect": []}
    seen: Dict[str, int] = {}

    for number, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        positions = [
            (line.find(":"), ":"),
            (line.find("="), "="),
        ]
        positions = [(position, marker) for position, marker in positions if position >= 0]
        if not positions:
            raise ProgramError(
                "%s:%d: expected 'key: value' or 'key = value'"
                % (source, number)
            )
        position, _marker = min(positions, key=lambda item: item[0])
        key = line[:position].strip().lower()
        value = line[position + 1 :].strip()
        if key not in _KNOWN_FIELDS:
            raise ProgramError(
                "%s:%d: unknown field %r; expected one of %s"
                % (source, number, key, ", ".join(sorted(_KNOWN_FIELDS)))
            )
        if key in _REPEAT_FIELDS:
            if not value:
                raise ProgramError("%s:%d: %s cannot be empty" % (source, number, key))
            values[key].append(value)
            continue
        if key in seen:
            raise ProgramError(
                "%s:%d: field %r already set on line %d"
                % (source, number, key, seen[key])
            )
        seen[key] = number
        values[key] = value

    return _build(
        values,
        base_dir=base_dir,
        source=source,
        text=text,
        source_format="text",
    )


def _parse_json(text: str, base_dir: Path, source: str) -> Dict[str, Any]:
    try:
        payload = json.loads(text)
    except ValueError as exc:
        raise ProgramError("%s: %s" % (source, exc)) from exc
    if not isinstance(payload, Mapping):
        raise ProgramError("%s: a JSON program must be an object" % source)
    unknown = sorted(set(payload) - _KNOWN_FIELDS - {"facts"})
    if unknown:
        raise ProgramError(
            "%s: unknown field(s): %s" % (source, ", ".join(unknown))
        )
    values: Dict[str, Any] = dict(payload)
    if "facts" in values:
        if "teach" in values:
            raise ProgramError("%s: use 'teach' or 'facts', not both" % source)
        values["teach"] = values.pop("facts")
    if isinstance(values.get("teach"), str):
        values["teach"] = [values["teach"]]
    values.setdefault("teach", [])
    values.setdefault("protect", [])
    return _build(
        values,
        base_dir=base_dir,
        source=source,
        text=text,
        source_format="json",
    )


def _build(
    values: Dict[str, Any],
    *,
    base_dir: Path,
    source: str,
    text: str,
    source_format: str,
) -> Dict[str, Any]:
    version = str(values.get("version", 1)).strip()
    if version != "1":
        raise ProgramError("%s: unsupported program version %r" % (source, version))

    model_value = values.get("model")
    output_value = values.get("output")
    if not isinstance(model_value, str) or not model_value.strip():
        raise ProgramError("%s: model is required" % source)
    if not isinstance(output_value, str) or not output_value.strip():
        raise ProgramError("%s: output is required" % source)
    model = _resolve_path(model_value, base_dir)
    output = _resolve_path(output_value, base_dir)
    if model == output:
        raise ProgramError("%s: model and output must be different paths" % source)

    teach_items = values.get("teach", [])
    if isinstance(teach_items, (str, Mapping)):
        teach_items = [teach_items]
    if not isinstance(teach_items, (list, tuple)) or not teach_items:
        raise ProgramError("%s: at least one teach entry is required" % source)
    facts = [
        _normalize_fact(item, index + 1)
        for index, item in enumerate(teach_items)
    ]
    protect = _normalize_protect(values.get("protect", []))

    budget = _positive_float(values.get("budget", 0.10), "budget", maximum=1.0)
    max_rel_norm = _positive_float(
        values.get("max_rel_norm", 0.02), "max_rel_norm", maximum=0.5
    )
    max_kl = _positive_float(values.get("max_kl", 0.05), "max_kl")
    try:
        min_panel_agreement = float(values.get("min_panel_agreement", 0.875))
    except (TypeError, ValueError) as exc:
        raise ProgramError("min_panel_agreement must be a number") from exc
    if not (0.0 <= min_panel_agreement <= 1.0):
        raise ProgramError("min_panel_agreement must be in [0, 1]")

    solver = str(values.get("solver", "direction")).strip().lower()
    if solver not in _SOLVERS:
        raise ProgramError(
            "solver must be one of %s" % ", ".join(sorted(_SOLVERS))
        )
    dtype = str(values.get("dtype", "float32")).strip().lower()
    if dtype not in _DTYPES:
        raise ProgramError("dtype must be one of %s" % ", ".join(sorted(_DTYPES)))
    quantize_value = values.get("quantize")
    quantize = None
    if quantize_value not in (None, ""):
        quantize = str(quantize_value).strip().lower()
        if quantize not in _QUANTIZE:
            raise ProgramError(
                "quantize must be one of %s" % ", ".join(sorted(_QUANTIZE))
            )
        if dtype != "auto":
            raise ProgramError("quantize requires dtype: auto")

    if values.get("fetch") not in (None, "") and values.get("starter") not in (None, ""):
        raise ProgramError("use fetch or starter, not both")
    fetch_value = values.get("fetch", values.get("starter"))
    fetch = None
    if fetch_value not in (None, ""):
        fetch = str(fetch_value).strip()
        from .fetch import RECOMMENDED

        if fetch not in RECOMMENDED:
            raise ProgramError(
                "unknown fetch starter %r; choose one of %s"
                % (fetch, ", ".join(sorted(RECOMMENDED)))
            )

    try:
        steps = int(values.get("steps", 250))
        max_layers = int(values.get("max_layers", 7))
    except (TypeError, ValueError) as exc:
        raise ProgramError("steps and max_layers must be integers") from exc
    lr = _positive_float(values.get("lr", 0.02), "lr")
    if steps < 1 or max_layers < 1:
        raise ProgramError("steps and max_layers must be positive")

    name = str(values.get("name", "")).strip()
    device_value = values.get("device")
    device = None if device_value in (None, "") else str(device_value).strip()

    return {
        "source_format": source_format,
        "path": source,
        "sha256": _sha256(text),
        "name": name,
        "model": model,
        "fetch": fetch,
        "output": output,
        "facts": facts,
        "protect": protect,
        "budget": budget,
        "max_rel_norm": max_rel_norm,
        "max_kl": max_kl,
        "min_panel_agreement": min_panel_agreement,
        "layers": parse_layers(values.get("layers")) or (),
        "solver": solver,
        "steps": steps,
        "lr": lr,
        "max_layers": max_layers,
        "dtype": dtype,
        "device": device,
        "quantize": quantize,
        "verify": _bool(values.get("verify", True), "verify"),
        "continue_on_reject": _bool(
            values.get("continue_on_reject", False), "continue_on_reject"
        ),
        "keep_steps": _bool(values.get("keep_steps", False), "keep_steps"),
    }


@dataclass(frozen=True)
class WeightProgram:
    """A validated, immutable weight-space program."""

    path: str
    sha256: str
    source_format: str
    name: str
    model: str
    fetch: Optional[str]
    output: str
    facts: Tuple[Dict[str, Any], ...]
    protect: Tuple[str, ...]
    budget: float
    max_rel_norm: float
    max_kl: float
    min_panel_agreement: float
    layers: Tuple[int, ...]
    solver: str
    steps: int
    lr: float
    max_layers: int
    dtype: str
    device: Optional[str]
    quantize: Optional[str]
    verify: bool
    continue_on_reject: bool
    keep_steps: bool

    def policy(self) -> TeachPolicy:
        return TeachPolicy(
            max_rel_norm=self.max_rel_norm,
            max_kl_p95=self.max_kl,
            min_panel_agreement=self.min_panel_agreement,
        )

    def pairs(self) -> List[Tuple[str, str]]:
        return [(fact["prompt"], fact["answer"]) for fact in self.facts]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": "machaon.program/1",
            "path": self.path,
            "sha256": self.sha256,
            "source_format": self.source_format,
            "name": self.name,
            "model": self.model,
            "fetch": self.fetch,
            "output": self.output,
            "facts": [dict(fact) for fact in self.facts],
            "protect": list(self.protect),
            "policy": {
                "budget": self.budget,
                "max_rel_norm": self.max_rel_norm,
                "max_kl_p95": self.max_kl,
                "min_panel_agreement": self.min_panel_agreement,
            },
            "runtime": {
                "solver": self.solver,
                "steps": self.steps,
                "lr": self.lr,
                "max_layers": self.max_layers,
                "layers": list(self.layers),
                "dtype": self.dtype,
                "device": self.device,
                "quantize": self.quantize,
            },
            "verify": self.verify,
            "continue_on_reject": self.continue_on_reject,
            "keep_steps": self.keep_steps,
        }


def parse_program(
    text: str,
    *,
    source: str = "<program>",
    base_dir: Optional[str | Path] = None,
) -> WeightProgram:
    """Parse and validate program text without loading a model."""
    if isinstance(text, str):
        text = text.lstrip("\ufeff")
    if not isinstance(text, str) or not text.strip():
        raise ProgramError("%s is empty" % source)
    base = Path(base_dir).expanduser().resolve() if base_dir is not None else Path.cwd()
    stripped = text.lstrip()
    if stripped.startswith("{"):
        values = _parse_json(text, base, source)
    else:
        values = _parse_text(text, base, source)
    return WeightProgram(**values)


def load_program(path: str | Path) -> WeightProgram:
    """Read a UTF-8 text or JSON program and resolve relative paths."""
    source = Path(path).expanduser().resolve()
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise ProgramError("%s: %s" % (source, exc)) from exc
    return parse_program(text, source=str(source), base_dir=source.parent)


def _one_line(value: Any, name: str, *, required: bool = False) -> str:
    text = "" if value is None else str(value).strip()
    if required and not text:
        raise ProgramError("%s cannot be empty" % name)
    if "\n" in text or "\r" in text:
        raise ProgramError("%s must be a single line" % name)
    return text


def _number(value: Any, name: str) -> str:
    number = _positive_float(value, name)
    return repr(number)


def _agreement(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ProgramError("min_panel_agreement must be a number") from exc
    if not math.isfinite(number) or not (0.0 <= number <= 1.0):
        raise ProgramError("min_panel_agreement must be in [0, 1]")
    return repr(number)


def render_program_text(
    *,
    model: str,
    output: str,
    teach: Sequence[str] | str,
    protect: Sequence[str] | str = (),
    fetch: Optional[str] = None,
    name: str = "",
    budget: float = 0.10,
    max_rel_norm: float = 0.02,
    max_kl: float = 0.05,
    min_panel_agreement: float = 0.875,
    layers: Any = None,
    solver: str = "direction",
    steps: int = 250,
    lr: float = 0.02,
    max_layers: int = 7,
    dtype: str = "float32",
    device: Optional[str] = None,
    quantize: Optional[str] = None,
    verify: bool = True,
    continue_on_reject: bool = False,
    keep_steps: bool = False,
) -> str:
    """Render a validated, reviewable text program.

    ``program-new`` and other callers use the same parser afterwards, so a
    generated file cannot introduce a second interpretation of the format.
    """
    model_text = _one_line(model, "model", required=True)
    output_text = _one_line(output, "output", required=True)
    name_text = _one_line(name, "name")
    fetch_text = _one_line(fetch, "fetch")
    device_text = _one_line(device, "device")
    if teach is None:
        teach_items = []
    elif isinstance(teach, str):
        teach_items = [teach]
    else:
        teach_items = list(teach)
    if protect is None:
        protect_items = []
    elif isinstance(protect, str):
        protect_items = [protect]
    else:
        protect_items = list(protect)
    if not teach_items:
        raise ProgramError("at least one teach entry is required")
    facts = [
        _one_line(item, "teach entry %d" % index, required=True)
        for index, item in enumerate(teach_items, 1)
    ]
    protected = [
        _one_line(item, "protect entry %d" % index, required=True)
        for index, item in enumerate(protect_items, 1)
    ]

    lines = [
        "# Generated by machaon program-new.",
        "# Review, then check: machaon program <this file> --plan",
        "version: 1",
    ]
    if name_text:
        lines.append("name: " + name_text)
    lines.append("model: " + model_text)
    if fetch_text:
        lines.append("fetch: " + fetch_text)
    lines.extend([
        "output: " + output_text,
        "budget: " + _number(budget, "budget"),
        "max_rel_norm: " + _number(max_rel_norm, "max_rel_norm"),
    ])
    if float(max_kl) != 0.05:
        lines.append("max_kl: " + _number(max_kl, "max_kl"))
    if float(min_panel_agreement) != 0.875:
        lines.append("min_panel_agreement: " + _agreement(min_panel_agreement))
    parsed_layers = parse_layers(layers)
    if parsed_layers:
        lines.append("layers: " + ",".join(str(item) for item in parsed_layers))
    if solver != "direction":
        lines.append("solver: " + _one_line(solver, "solver", required=True))
    if int(steps) != 250:
        lines.append("steps: %d" % int(steps))
    if float(lr) != 0.02:
        lines.append("lr: " + _number(lr, "lr"))
    if int(max_layers) != 7:
        lines.append("max_layers: %d" % int(max_layers))
    if dtype != "float32":
        lines.append("dtype: " + _one_line(dtype, "dtype", required=True))
    if device_text:
        lines.append("device: " + device_text)
    if quantize:
        lines.append("quantize: " + _one_line(quantize, "quantize"))
    lines.append("verify: " + ("true" if verify else "false"))
    if continue_on_reject:
        lines.append("continue_on_reject: true")
    if keep_steps:
        lines.append("keep_steps: true")
    lines.append("")
    lines.extend("teach: " + item for item in facts)
    if protected:
        lines.append("")
        lines.extend("protect: " + item for item in protected)
    text = "\n".join(lines) + "\n"
    parse_program(text, source="<generated program>", base_dir=Path.cwd())
    return text


@dataclass
class ProgramRun:
    """A completed program run and its persisted-byte verification."""

    program: Dict[str, Any]
    run: Dict[str, Any]
    verification: Optional[Dict[str, Any]]
    status: str
    accepted: bool
    summary: str
    report_path: Optional[str]
    report_sha256: Optional[str] = None
    model_fetched: bool = False
    effective_output: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": "machaon.program/1",
            "program": dict(self.program),
            "run": dict(self.run),
            "verification": self.verification,
            "status": self.status,
            "accepted": bool(self.accepted),
            "summary": self.summary,
            "report_path": self.report_path,
            "report_format": (
                PROGRAM_REPORT_FORMAT if self.report_sha256 else None
            ),
            "report_format_version": (
                PROGRAM_REPORT_VERSION if self.report_sha256 else None
            ),
            "report_sha256": self.report_sha256,
            "model_fetched": bool(self.model_fetched),
            "effective_output": self.effective_output,
        }


def _as_dict(value: Any) -> Dict[str, Any]:
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError("expected a mapping or an object with to_dict()")


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProgramReportError(
            "program report is not JSON-serialisable: %s" % exc
        ) from exc


def program_report_sha256(document: Mapping[str, Any]) -> str:
    """Address a program report excluding the address field itself."""
    if not isinstance(document, Mapping):
        raise ProgramReportError("program report must be an object")
    body = {key: value for key, value in document.items()
            if key != "report_sha256"}
    return hashlib.sha256(_canonical_bytes(body)).hexdigest()


def write_program_report(
    path: str | Path, payload: Mapping[str, Any]
) -> Dict[str, Any]:
    """Atomically write a self-addressed program report and return it."""
    if not isinstance(payload, Mapping):
        raise ProgramReportError("program report must be an object")
    document = dict(payload)
    document.pop("report_sha256", None)
    document["report_format"] = PROGRAM_REPORT_FORMAT
    document["report_format_version"] = PROGRAM_REPORT_VERSION
    document["report_sha256"] = program_report_sha256(document)

    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=target.parent,
        prefix=".program-report-",
        suffix=".tmp",
        delete=False,
        encoding="utf-8",
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(
            document,
            handle,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
    try:
        os.replace(temp_path, target)
    finally:
        temp_path.unlink(missing_ok=True)
    return document


def parse_program_report(
    text: str, *, source: str = "<program report>"
) -> Dict[str, Any]:
    """Parse and integrity-check one report without loading a model."""
    try:
        document = json.loads(text)
    except ValueError as exc:
        raise ProgramReportError("%s: %s" % (source, exc)) from exc
    if not isinstance(document, dict):
        raise ProgramReportError("%s: a program report must be an object" % source)
    if document.get("report_format") != PROGRAM_REPORT_FORMAT:
        raise ProgramReportError(
            "%s: unsupported report format %r"
            % (source, document.get("report_format"))
        )
    if document.get("report_format_version") != PROGRAM_REPORT_VERSION:
        raise ProgramReportError(
            "%s: unsupported report version %r"
            % (source, document.get("report_format_version"))
        )
    address = document.get("report_sha256")
    if not isinstance(address, str) or len(address) != 64:
        raise ProgramReportError("%s: report is missing its SHA-256 address" % source)
    try:
        actual = program_report_sha256(document)
    except ProgramReportError as exc:
        raise ProgramReportError("%s: %s" % (source, exc)) from exc
    if address != actual:
        raise ProgramReportError("%s: report address mismatch" % source)
    program = document.get("program")
    if not isinstance(program, dict):
        raise ProgramReportError("%s: report needs a program object" % source)
    sha256 = program.get("sha256")
    if not isinstance(sha256, str) or len(sha256) != 64:
        raise ProgramReportError(
            "%s: report program is missing its SHA-256 address" % source
        )
    if not isinstance(program.get("facts"), list):
        raise ProgramReportError("%s: report program needs a facts list" % source)
    if not isinstance(document.get("run"), dict):
        raise ProgramReportError("%s: report needs a run object" % source)
    return document


def read_program_report(path: str | Path) -> Dict[str, Any]:
    """Read and integrity-check a saved program report."""
    source = Path(path).expanduser()
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise ProgramReportError("%s: %s" % (source, exc)) from exc
    return parse_program_report(text, source=str(source))


def _report_facts(document: Mapping[str, Any]) -> List[Tuple[str, str]]:
    program = document.get("program")
    facts = program.get("facts") if isinstance(program, Mapping) else None
    if not isinstance(facts, list) or not facts:
        raise ProgramReportError("program report has no facts to check")
    pairs: List[Tuple[str, str]] = []
    for index, fact in enumerate(facts, 1):
        if not isinstance(fact, Mapping):
            raise ProgramReportError("program report fact %d is not an object" % index)
        prompt = fact.get("prompt")
        answer = fact.get("answer")
        if not isinstance(prompt, str) or not isinstance(answer, str):
            raise ProgramReportError(
                "program report fact %d needs prompt and answer strings" % index
            )
        pairs.append((prompt, answer))
    return pairs


def program_report_checkpoint(document: Mapping[str, Any]) -> str:
    """Return the checkpoint named by a report, or explain why there is none."""
    run = document.get("run")
    if isinstance(run, Mapping):
        final = run.get("final")
        if isinstance(final, str) and final.strip():
            return final
    if document.get("status") == "no_edits":
        program = document.get("program")
        model = program.get("model") if isinstance(program, Mapping) else None
        if isinstance(model, str) and model.strip():
            return model
    raise ProgramReportError(
        "program report has no saved checkpoint to check; "
        "the run did not produce a final checkpoint"
    )


def verify_program_report(
    report: str | Path | Mapping[str, Any],
    *,
    program_path: Optional[str | Path] = None,
    check_model: bool = False,
    checkpoint: Optional[str | Path] = None,
    check_fn: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Verify report integrity, its program address, and optionally its facts.

    Integrity verification is local and needs no model.  ``check_model=True``
    reloads the checkpoint named by the report (or ``checkpoint`` when the
    saved bytes have moved) and re-runs every declared fact; that is the check
    that turns a valid report into fresh behavioural evidence.
    """
    if isinstance(report, Mapping):
        document = parse_program_report(
            json.dumps(report, ensure_ascii=False),
            source="<inline program report>",
        )
    elif isinstance(report, (str, Path)):
        document = read_program_report(report)
    else:
        raise ProgramReportError("report must be a path or a JSON object")

    program = document["program"]
    result: Dict[str, Any] = {
        "valid": True,
        "format": PROGRAM_REPORT_FORMAT,
        "format_version": PROGRAM_REPORT_VERSION,
        "report_sha256": document["report_sha256"],
        "program_sha256": program["sha256"],
        "program_name": program.get("name", ""),
        "status": document.get("status"),
        "accepted": bool(document.get("accepted")),
        "facts": len(program.get("facts", [])),
        "program_hash_matches": None,
        "model_check": None,
    }

    if program_path is not None:
        source = load_program(program_path)
        if source.sha256 != program["sha256"]:
            raise ProgramReportError(
                "program hash does not match the report: expected %s, got %s"
                % (program["sha256"], source.sha256)
            )
        result["program_hash_matches"] = True
        result["program_path"] = source.path

    if check_model:
        target = (
            str(Path(checkpoint).expanduser())
            if checkpoint is not None
            else program_report_checkpoint(document)
        )
        runtime = program.get("runtime")
        runtime = runtime if isinstance(runtime, Mapping) else {}
        pairs = _report_facts(document)
        checker = check_fn
        if checker is None:
            from .check import check_facts

            checker = check_facts
        try:
            checked = _as_dict(checker(
                target,
                pairs,
                dtype=str(runtime.get("dtype", "auto")),
                device=runtime.get("device"),
                quantize=runtime.get("quantize"),
            ))
        except Exception as exc:
            raise ProgramReportError(
                "saved-byte check could not run: %s: %s"
                % (type(exc).__name__, exc)
            ) from exc
        live = bool(checked.get("passed"))
        reported = document.get("verification")
        reported_passed = (
            bool(reported.get("passed"))
            if isinstance(reported, Mapping)
            else None
        )
        result["model_check"] = {
            "performed": True,
            "checkpoint": target,
            "passed": live,
            "reported_passed": reported_passed,
            "matches_report": (
                None if reported_passed is None else live == reported_passed
            ),
            "facts": checked.get("facts", []),
        }
    return result


def run_program(
    program: WeightProgram | str | Path,
    *,
    out_override: Optional[str | Path] = None,
    verify: Optional[bool] = None,
    fetch_fn: Optional[Callable[..., Mapping[str, Any]]] = None,
    teach_many_fn: Optional[Callable[..., Any]] = None,
    check_fn: Optional[Callable[..., Any]] = None,
    session_factory: Optional[Callable[[str, EditBudget], Any]] = None,
    progress: Optional[Callable[[Mapping[str, Any]], None]] = None,
    report_out: Optional[str | Path] = None,
) -> ProgramRun:
    """Run a program, save its checkpoint chain, and verify the saved bytes.

    The default path is the same ``teach_many`` chain used by the CLI.  The
    program layer only adds path resolution, declared policy, a persisted
    ``program-report.json``, and a reload check on the final checkpoint.
    """
    if not isinstance(program, WeightProgram):
        program = load_program(program)

    from .fetch import RECOMMENDED, fetch_model

    model = Path(program.model)
    output = (
        Path(out_override).expanduser().resolve()
        if out_override is not None
        else Path(program.output)
    )
    report_path = (
        Path(report_out).expanduser().resolve()
        if report_out is not None
        else output / PROGRAM_REPORT_FILENAME
    )
    if report_out is not None and report_path.exists():
        raise FileExistsError(
            "report path already exists: %s" % report_path
        )
    if str(output) == str(model):
        raise ProgramError("output and model must be different paths")
    try:
        output.relative_to(model)
    except ValueError:
        pass
    else:
        raise ProgramError("output must not be inside the model directory")
    try:
        model.relative_to(output)
    except ValueError:
        pass
    else:
        raise ProgramError("model must not be inside the output directory")
    if output.exists() and not output.is_dir():
        raise ProgramError("output exists and is not a directory: %s" % output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            "output directory already exists and is not empty: %s" % output
        )

    fetched = False
    if not model.exists():
        if program.fetch is None:
            raise ProgramError(
                "%s does not exist; add 'fetch: smol135m' or download it first"
                % model
            )
        starter = RECOMMENDED.get(program.fetch)
        if starter is None:
            raise ProgramError(
                "unknown fetch starter %r; choose one of %s"
                % (program.fetch, ", ".join(sorted(RECOMMENDED)))
            )
        fetch = fetch_fn or fetch_model
        fetch(starter["repo"], model, require_new=False)
        fetched = True
    if not model.is_dir():
        raise ProgramError("model must be a checkpoint directory: %s" % model)
    # ``teach_many`` accepts an existing empty directory, which makes it
    # possible to persist the program report even when the chain stops before
    # its first fact.  A non-empty directory was refused above.
    output.mkdir(parents=True, exist_ok=True)

    if session_factory is None:
        def session_factory(path: str, step_budget: EditBudget):
            from .session import Session

            return Session.open(
                str(path),
                dtype=program.dtype,
                device=program.device,
                quantize=program.quantize,
                budget=step_budget,
            )

    if teach_many_fn is None:
        from .teach_many import teach_many as teach_many_fn

    try:
        kwargs = {
            "budget": program.budget,
            "policy": program.policy(),
            "preserve_prompts": list(program.protect) or None,
            "layers": list(program.layers) or None,
            "solver": program.solver,
            "steps": program.steps,
            "lr": program.lr,
            "max_layers": program.max_layers,
            "keep_steps": program.keep_steps,
            "continue_on_reject": program.continue_on_reject,
        }
        if progress is not None:
            kwargs["progress"] = progress
        run = _as_dict(teach_many_fn(
            session_factory,
            str(model),
            program.pairs(),
            output,
            **kwargs,
        ))
    except Exception as exc:
        run = {
            "status": "error",
            "final": None,
            "out_dir": str(output),
            "facts": len(program.facts),
            "summary": str(exc),
            "error": "%s: %s" % (type(exc).__name__, exc),
        }
    steps = run.get("steps", [])
    if isinstance(steps, list):
        run.setdefault("applied", sum(
            1 for step in steps
            if isinstance(step, Mapping) and step.get("status") == "applied"
        ))
        run.setdefault("rejected", sum(
            1 for step in steps
            if isinstance(step, Mapping) and step.get("status") == "rejected"
        ))
        run.setdefault("already_true", sum(
            1 for step in steps
            if isinstance(step, Mapping) and step.get("status") == "already_true"
        ))

    do_verify = program.verify if verify is None else bool(verify)
    verification: Optional[Dict[str, Any]] = None
    verification_target = run.get("final")
    if verification_target is None and run.get("status") == "no_edits":
        verification_target = str(model)
    if do_verify and verification_target is not None:
        checker = check_fn
        if checker is None:
            from .check import check_facts

            checker = check_facts
        try:
            verification = _as_dict(checker(
                verification_target,
                program.pairs(),
                dtype=program.dtype,
                device=program.device,
                quantize=program.quantize,
            ))
        except Exception as exc:
            verification = {
                "passed": False,
                "facts": [],
                "error": "%s: %s" % (type(exc).__name__, exc),
            }

    run_ok = run.get("status") in ("applied", "no_edits")
    verification_ok = verification is None or bool(verification.get("passed"))
    if run_ok and do_verify and verification is None:
        status = "verification_skipped"
    elif run_ok and verification_ok:
        status = str(run.get("status"))
    elif run_ok:
        status = "verification_failed"
    else:
        status = str(run.get("status", "error"))

    if status == "applied":
        applied_count = int(run.get("applied", len(program.facts)))
        if verification is not None:
            facts = verification.get("facts", [])
            passed = sum(1 for item in facts if item.get("passed"))
            summary = (
                "Applied %d fact(s) and saved %s. Reloaded the saved bytes: "
                "%d/%d facts pass."
                % (
                    applied_count,
                    run.get("final") or output,
                    passed,
                    len(facts),
                )
            )
        else:
            summary = (
                "Applied %d fact(s) and saved %s. Saved-byte verification was "
                "disabled." % (applied_count, run.get("final") or output)
            )
    elif status == "no_edits":
        summary = (
            "No edit was needed; the base checkpoint already answered every "
            "fact. %s" % run.get("summary", "")
        ).strip()
    elif status == "verification_failed":
        failed = [
            item for item in (verification or {}).get("facts", [])
            if not item.get("passed")
        ]
        summary = (
            "The checkpoint chain finished, but %d saved fact(s) failed "
            "verification. Inspect %s before using the output."
            % (len(failed), run.get("final") or output)
        )
    else:
        summary = str(run.get("summary", "The program did not complete."))

    payload = {
        "format": "machaon.program/1",
        "program": program.to_dict(),
        "run": run,
        "verification": verification,
        "status": status,
        "accepted": bool(run_ok and verification_ok and status != "verification_failed"),
        "summary": summary,
        "report_path": str(report_path),
        "model_fetched": fetched,
        "effective_output": str(output),
    }
    report_sha256: Optional[str] = None
    try:
        output.mkdir(parents=True, exist_ok=True)
        written = write_program_report(report_path, payload)
        report_sha256 = str(written["report_sha256"])
    except (OSError, TypeError, ValueError):
        report_path = None

    return ProgramRun(
        program=program.to_dict(),
        run=run,
        verification=verification,
        status=status,
        accepted=bool(payload["accepted"]),
        summary=summary,
        report_path=str(report_path) if report_path else None,
        report_sha256=report_sha256,
        model_fetched=fetched,
        effective_output=str(output),
    )
