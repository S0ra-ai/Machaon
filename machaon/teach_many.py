"""Teach several facts into one checkpoint chain.

Each fact runs through the same :func:`machaon.recipes.teach_fact` loop on a
fresh session loaded from the previous accepted checkpoint.  That keeps every
step honest: the layer search starts from a clean model, the per-fact checks
measure that step, and the original checkpoint is never modified.

The chain also tracks two things a single edit cannot see:

* the cumulative relative change against the session budget, and
* top-1 agreement and KL on the original baseline panel, so a sequence of
  individually acceptable edits cannot silently drift the model away.

If a fact is rejected or the cumulative panel check fails, the last safe
checkpoint is kept and the failed step is removed (unless ``keep_steps``).
"""
from __future__ import annotations

import gc
import json
import os
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .budget import EditBudget
from .checkpointing import save_session
from .hygieia.metrics import kl_summary, softmax
from .intent import IntentError, parse_intent
from .recipes import DEFAULT_BASELINE_PROMPTS, TeachPolicy, TeachResult, teach_fact

__all__ = [
    "TeachStep",
    "TeachManyResult",
    "default_session_factory",
    "load_facts",
    "teach_many",
]

Fact = Tuple[str, str]


def _normalize_fact(item: Any, index: int) -> Fact:
    if isinstance(item, str):
        try:
            intent = parse_intent(item)
        except IntentError as exc:
            raise ValueError("fact %d: %s" % (index, exc)) from exc
        return intent.prompt, intent.answer
    if isinstance(item, Mapping):
        if "say" in item:
            try:
                intent = parse_intent(str(item["say"]))
            except IntentError as exc:
                raise ValueError("fact %d: %s" % (index, exc)) from exc
            return intent.prompt, intent.answer
        prompt = item.get("prompt")
        answer = item.get("answer")
        if isinstance(prompt, str) and isinstance(answer, str) and prompt.strip():
            return prompt, answer
        raise ValueError(
            "fact %d needs 'say', or string 'prompt' and 'answer'" % index
        )
    if isinstance(item, (list, tuple)) and len(item) == 2:
        prompt, answer = item
        if isinstance(prompt, str) and isinstance(answer, str) and prompt.strip():
            return prompt, answer
    raise ValueError("fact %d must be a sentence, a pair, or an object" % index)


def load_facts(path: str | Path) -> List[Fact]:
    """Load facts from a text, JSON or JSONL file.

    Text files use one teaching sentence per line (``prompt -> answer``,
    labelled blocks, or any grammar accepted by :func:`machaon.intent.parse_intent`).
    JSON files accept a list of strings, a list of ``{"say": ...}`` objects, a
    list of ``{"prompt": ..., "answer": ...}`` objects, or an object with a
    ``facts`` list.  JSONL accepts one such item per line.
    """
    source = Path(path)
    raw = source.read_text(encoding="utf-8")
    if not raw.strip():
        raise ValueError("%s is empty" % source)
    suffix = source.suffix.lower()
    stripped = raw.strip()

    def from_payload(payload: Any) -> List[Fact]:
        if isinstance(payload, Mapping) and "facts" in payload:
            payload = payload["facts"]
        if isinstance(payload, Mapping):
            payload = [payload]
        if not isinstance(payload, list):
            raise ValueError("facts JSON must be a list or an object with 'facts'")
        if not payload:
            raise ValueError("facts JSON is empty")
        return [_normalize_fact(item, index + 1) for index, item in enumerate(payload)]

    if suffix == ".jsonl":
        items = []
        for number, line in enumerate(raw.splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                items.append(json.loads(line))
            except ValueError as exc:
                raise ValueError("line %d: %s" % (number, exc)) from exc
        if not items:
            raise ValueError("%s contains no facts" % source)
        return [_normalize_fact(item, index + 1) for index, item in enumerate(items)]

    if suffix == ".json" or stripped[0] in "[{":
        try:
            payload = json.loads(stripped)
        except ValueError as exc:
            raise ValueError("%s: %s" % (source, exc)) from exc
        return from_payload(payload)

    facts: List[Fact] = []
    for number, line in enumerate(raw.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            intent = parse_intent(line)
        except IntentError as exc:
            raise ValueError("line %d: %s" % (number, exc)) from exc
        facts.append((intent.prompt, intent.answer))
    if not facts:
        raise ValueError("%s contains no facts" % source)
    return facts


def default_session_factory(path: str, budget: EditBudget):
    from .session import Session

    return Session.open(str(path), budget=budget)


@dataclass
class TeachStep:
    index: int
    prompt: str
    answer: str
    status: str
    checkpoint: Optional[str] = None
    kept: bool = False
    cumulative_relative_delta: float = 0.0
    global_panel_agreement: Optional[float] = None
    global_kl_p95: Optional[float] = None
    global_passed: Optional[bool] = None
    result: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": int(self.index),
            "prompt": self.prompt,
            "answer": self.answer,
            "status": self.status,
            "checkpoint": self.checkpoint,
            "kept": bool(self.kept),
            "cumulative_relative_delta": float(self.cumulative_relative_delta),
            "global_panel_agreement": self.global_panel_agreement,
            "global_kl_p95": self.global_kl_p95,
            "global_passed": self.global_passed,
            "result": self.result,
            "error": self.error,
        }


@dataclass
class TeachManyResult:
    status: str
    base_model: str
    out_dir: str
    final: Optional[str]
    facts: int
    budget: float
    cumulative_relative_delta: float
    panel_size: int
    global_panel_agreement: Optional[float]
    global_kl_p95: Optional[float]
    global_passed: Optional[bool]
    stopped_at: Optional[int]
    reason: str
    summary: str
    steps: List[TeachStep] = field(default_factory=list)
    report_path: Optional[str] = None

    @property
    def accepted(self) -> bool:
        return self.status in ("applied", "no_edits")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "accepted": bool(self.accepted),
            "base_model": self.base_model,
            "out_dir": self.out_dir,
            "final": self.final,
            "facts": int(self.facts),
            "budget": float(self.budget),
            "cumulative_relative_delta": float(self.cumulative_relative_delta),
            "panel_size": int(self.panel_size),
            "global_panel_agreement": self.global_panel_agreement,
            "global_kl_p95": self.global_kl_p95,
            "global_passed": self.global_passed,
            "stopped_at": self.stopped_at,
            "reason": self.reason,
            "summary": self.summary,
            "steps": [step.to_dict() for step in self.steps],
            "report_path": self.report_path,
        }


def _global_check(
    current_logits: Any,
    original_logits: Any,
    original_top1: Any,
    policy: TeachPolicy,
) -> Tuple[float, float, bool]:
    current = np.asarray(current_logits)
    original = np.asarray(original_logits)
    agreement = float((current.argmax(axis=-1) == original_top1).mean())
    kl = kl_summary(softmax(original), softmax(current))
    kl_p95 = float(kl["kl_p95"])
    passed = (
        agreement >= policy.min_panel_agreement
        and kl_p95 <= policy.max_kl_p95
    )
    return agreement, kl_p95, passed


def _summary(
    status: str,
    applied: int,
    total: int,
    final: Optional[str],
    cumulative: float,
    panel_size: int,
    agreement: Optional[float],
    kl_p95: Optional[float],
    reason: str,
) -> str:
    final_text = "no checkpoint (source unchanged)" if not final else final
    if applied == 0:
        return (
            "No facts were applied. %s Source checkpoint: %s"
            % (reason or "", final_text)
        )
    panel = (
        "global panel not measured" if agreement is None
        else "%d/%d top-1, KL p95 %.3g" % (
            round(agreement * panel_size), panel_size, kl_p95,
        )
    )
    return (
        "Applied %d of %d fact(s). Final checkpoint: %s. Cumulative Δ %.4f; "
        "%s.%s"
        % (
            applied,
            total,
            final_text,
            cumulative,
            panel,
            "" if not reason else " " + reason,
        )
    )


def teach_many(
    session_factory: Callable[[str, EditBudget], Any],
    base_model: str,
    facts: Sequence[Any],
    out_dir: str | Path,
    *,
    budget: float = 0.10,
    policy: Optional[TeachPolicy] = None,
    preserve_prompts: Optional[Sequence[str]] = None,
    layers: Optional[Sequence[int]] = None,
    solver: str = "direction",
    steps: int = 250,
    lr: float = 0.02,
    max_layers: int = 7,
    keep_steps: bool = False,
    continue_on_reject: bool = False,
    min_step: float = 0.001,
    teach_one: Callable[..., TeachResult] = teach_fact,
    save: Callable[[Any, str | Path], Mapping[str, Any]] = save_session,
    progress: Optional[Callable[[Mapping[str, Any]], None]] = None,
) -> TeachManyResult:
    """Teach a sequence of facts into a chain of checkpoint directories."""
    from .checkpointing import checkpoint_problem

    problem = checkpoint_problem(base_model)
    if problem:
        raise ValueError("%s: %s" % (base_model, problem))
    budget = float(budget)
    if budget <= 0:
        raise ValueError("budget must be positive")
    policy = policy or TeachPolicy()
    normalized = [
        _normalize_fact(item, index + 1)
        for index, item in enumerate(list(facts))
    ]
    if not normalized:
        raise ValueError("teach_many needs at least one fact")
    out = Path(out_dir).expanduser()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(
            "output directory already exists and is not empty: %s" % out
        )
    out.mkdir(parents=True, exist_ok=True)
    def report(**payload: Any) -> None:
        if progress is not None:
            try:
                progress(payload)
            except Exception:
                # Progress is observational and must never change the edit.
                pass

    report(phase="program", status="started", total=len(normalized), out_dir=str(out))

    panel = [
        str(item) for item in (
            list(preserve_prompts)
            if preserve_prompts is not None
            else list(DEFAULT_BASELINE_PROMPTS)
        )
        if isinstance(item, str) and item.strip()
    ]
    panel = list(dict.fromkeys(panel))
    if not panel and policy.require_panel:
        raise ValueError("at least one preserve prompt is required")

    current_model = str(base_model)
    cumulative = 0.0
    original_logits = None
    original_top1 = None
    last_safe: Optional[Path] = None
    last_safe_index: Optional[int] = None
    last_agreement: Optional[float] = None
    last_kl: Optional[float] = None
    last_global_passed: Optional[bool] = None
    steps_out: List[TeachStep] = []
    applied = 0
    status = "no_edits"
    stopped_at: Optional[int] = None
    reason = ""
    session = None

    for index, (prompt, answer) in enumerate(normalized, 1):
        report(
            phase="program",
            status="teaching",
            step=index,
            total=len(normalized),
            prompt=prompt,
        )
        remaining = budget - cumulative
        if remaining < min_step:
            status = "budget_exhausted"
            stopped_at = index
            reason = "cumulative budget leaves %.4f, below the minimum step" % remaining
            break

        step_policy = replace(
            policy, max_rel_norm=min(policy.max_rel_norm, remaining)
        )
        try:
            session = session_factory(current_model, EditBudget(max_total=remaining))
            if original_logits is None and panel:
                original_logits = np.asarray(session.runtime.next_logits(panel))
                original_top1 = original_logits.argmax(axis=-1)
            result = teach_one(
                session,
                prompt,
                answer,
                layers=layers,
                policy=step_policy,
                solver=solver,
                steps=steps,
                lr=lr,
                max_layers=max_layers,
                preserve_prompts=panel,
            )
        except ImportError:
            # A missing optional runtime is a configuration error, not a fact
            # that failed its checks. Let the caller report it once instead of
            # burying it in a per-fact step.
            raise
        except Exception as exc:
            steps_out.append(TeachStep(
                index=index,
                prompt=prompt,
                answer=answer,
                status="error",
                cumulative_relative_delta=cumulative,
                error="%s: %s" % (type(exc).__name__, exc),
            ))
            status = "error"
            stopped_at = index
            reason = "fact %d raised %s" % (index, exc)
            session = None
            gc.collect()
            break

        payload = result.to_dict()
        step = TeachStep(
            index=index,
            prompt=prompt,
            answer=answer,
            status=result.status,
            cumulative_relative_delta=cumulative,
            result=payload,
        )

        if result.status == "applied":
            step_path = out / ("step-%03d" % index)
            try:
                saved = save(session, step_path)
            except Exception as exc:
                step.status = "error"
                step.error = "%s: %s" % (type(exc).__name__, exc)
                steps_out.append(step)
                status = "error"
                stopped_at = index
                reason = "saving fact %d failed: %s" % (index, exc)
                session = None
                gc.collect()
                break
            saved_output = Path(saved.get("output", step_path))
            step.checkpoint = str(saved_output)
            step.kept = True

            if panel and original_logits is not None:
                try:
                    current = session.runtime.next_logits(panel)
                    agreement, kl_p95, global_passed = _global_check(
                        current, original_logits, original_top1, policy
                    )
                except Exception as exc:
                    step.status = "error"
                    step.error = "global panel check failed: %s" % exc
                    steps_out.append(step)
                    status = "error"
                    stopped_at = index
                    reason = "global panel check failed at fact %d: %s" % (index, exc)
                    if not keep_steps:
                        shutil.rmtree(saved_output, ignore_errors=True)
                        step.kept = False
                    session = None
                    gc.collect()
                    break
                step.global_panel_agreement = agreement
                step.global_kl_p95 = kl_p95
                step.global_passed = global_passed
                if not global_passed:
                    status = "global_reject"
                    stopped_at = index
                    reason = (
                        "cumulative drift failed at fact %d: %d/%d top-1, "
                        "KL p95 %.3g" % (
                            index,
                            round(agreement * len(panel)),
                            len(panel),
                            kl_p95,
                        )
                    )
                    if not keep_steps:
                        shutil.rmtree(saved_output, ignore_errors=True)
                        step.kept = False
                    steps_out.append(step)
                    session = None
                    gc.collect()
                    break

            cumulative += float(result.relative_delta or 0.0)
            step.cumulative_relative_delta = cumulative
            last_agreement = step.global_panel_agreement
            last_kl = step.global_kl_p95
            last_global_passed = step.global_passed
            previous_safe = last_safe
            last_safe = saved_output
            last_safe_index = index
            applied += 1
            status = "applied"
            steps_out.append(step)

            if previous_safe is not None and not keep_steps:
                shutil.rmtree(previous_safe, ignore_errors=True)
                for earlier in steps_out:
                    if earlier.checkpoint == str(previous_safe):
                        earlier.kept = False
            current_model = str(saved_output)
            report(
                phase="program",
                status=step.status,
                step=index,
                total=len(normalized),
                prompt=prompt,
                layer=step.result.get("best_layer"),
                delta=step.cumulative_relative_delta,
                global_passed=step.global_passed,
            )

        elif result.status == "already_true":
            steps_out.append(step)
            report(
                phase="program",
                status=step.status,
                step=index,
                total=len(normalized),
                prompt=prompt,
            )
            session = None
            gc.collect()
            continue
        elif result.status == "rejected":
            steps_out.append(step)
            report(
                phase="program",
                status=step.status,
                step=index,
                total=len(normalized),
                prompt=prompt,
            )
            if continue_on_reject:
                session = None
                gc.collect()
                continue
            status = "stopped"
            stopped_at = index
            reason = "fact %d was rejected; the last safe checkpoint was kept" % index
            session = None
            gc.collect()
            break

        else:
            steps_out.append(step)
            report(
                phase="program",
                status="error",
                step=index,
                total=len(normalized),
                prompt=prompt,
            )
            status = "error"
            stopped_at = index
            reason = "fact %d returned status %r" % (index, result.status)
            session = None
            gc.collect()
            break

        session = None
        gc.collect()

    final: Optional[Path] = None
    if last_safe is not None:
        if keep_steps:
            final = last_safe
        else:
            final_dir = out / "final"
            if final_dir.exists():
                shutil.rmtree(final_dir, ignore_errors=True)
            os.replace(last_safe, final_dir)
            final = final_dir
            if last_safe_index is not None:
                for step in steps_out:
                    if step.index == last_safe_index:
                        step.checkpoint = str(final)
    elif status == "applied":
        status = "no_edits"
    if (
        status == "no_edits"
        and steps_out
        and all(step.status == "already_true" for step in steps_out)
    ):
        reason = "every fact was already true; no edit was needed"

    report(
        phase="program",
        status="finished",
        outcome=status,
        total=len(normalized),
        applied=applied,
        final=None if final is None else str(final),
        cumulative_delta=cumulative,
    )

    result = TeachManyResult(
        status=status,
        base_model=str(base_model),
        out_dir=str(out),
        final=None if final is None else str(final),
        facts=len(normalized),
        budget=budget,
        cumulative_relative_delta=cumulative,
        panel_size=len(panel),
        global_panel_agreement=last_agreement,
        global_kl_p95=last_kl,
        global_passed=last_global_passed,
        stopped_at=stopped_at,
        reason=reason,
        summary=_summary(
            status,
            applied,
            len(normalized),
            None if final is None else str(final),
            cumulative,
            len(panel),
            last_agreement,
            last_kl,
            reason,
        ),
        steps=steps_out,
    )
    report_path = out / "report.json"
    result.report_path = str(report_path)
    report_path.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return result
