"""The zero-to-one first edit.

``machaon first`` is deliberately a thin orchestration over the existing
pieces: fetch a starter if the checkpoint is absent, open it, run the one-call
teaching recipe, and save the accepted result.  It adds no new edit semantics.
"""
from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from .budget import EditBudget
from .checkpointing import save_session
from .fetch import RECOMMENDED, fetch_model
from .intent import parse_intent
from .recipes import TeachPolicy, teach_fact

__all__ = ["FirstResult", "first_edit"]

_RUNTIME_PACKAGES = ("torch", "transformers", "safetensors")


def missing_runtime_packages() -> Sequence[str]:
    """Names of the optional real-model packages this interpreter cannot import."""
    from importlib.util import find_spec

    missing = []
    for name in _RUNTIME_PACKAGES:
        try:
            if find_spec(name) is None:
                missing.append(name)
        except (ImportError, ValueError):
            missing.append(name)
    return tuple(missing)


def _as_dict(value: Any) -> Dict[str, Any]:
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    if isinstance(value, Mapping):
        return dict(value)
    return {"passed": bool(getattr(value, "passed", False))}


@dataclass
class FirstResult:
    status: str
    model: str
    model_fetched: bool
    sentence: str
    prompt: str
    answer: str
    final: Optional[str]
    teach: Dict[str, Any]
    summary: str
    verification: Optional[Dict[str, Any]] = None

    @property
    def accepted(self) -> bool:
        return self.status in ("applied", "already_true")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "accepted": bool(self.accepted),
            "model": self.model,
            "model_fetched": bool(self.model_fetched),
            "sentence": self.sentence,
            "prompt": self.prompt,
            "answer": self.answer,
            "final": self.final,
            "teach": dict(self.teach),
            "summary": self.summary,
            "verification": self.verification,
        }


def first_edit(
    sentence: str,
    model: str | Path,
    out: Optional[str | Path] = None,
    *,
    starter: str = "smol135m",
    budget: float = 0.10,
    policy: Optional[TeachPolicy] = None,
    layers: Optional[Sequence[int]] = None,
    solver: str = "direction",
    steps: int = 250,
    lr: float = 0.02,
    max_layers: int = 7,
    dtype: str = "float32",
    device: Optional[str] = None,
    quantize: Optional[str] = None,
    no_save: bool = False,
    fetch_fn: Callable[..., Mapping[str, Any]] = fetch_model,
    teach_one: Callable[..., Any] = teach_fact,
    save_fn: Callable[..., Mapping[str, Any]] = save_session,
    session_factory: Optional[Callable[[str, EditBudget], Any]] = None,
    verify_saved: bool = True,
    check_fn: Optional[Callable[..., Any]] = None,
) -> FirstResult:
    """Download if needed, teach one sentence, and save the accepted edit."""
    intent = parse_intent(sentence)
    real_runtime = session_factory is None
    if session_factory is None:
        missing = missing_runtime_packages()
        if missing:
            raise ImportError(
                'real-model support is not installed; pip install "machaon[hf]" '
                "(missing: %s)" % ", ".join(missing)
            )
    model_path = Path(model).expanduser()
    fetched = False
    if not model_path.exists():
        item = RECOMMENDED.get(starter)
        if item is None:
            raise ValueError(
                "unknown starter %r; choose one of %s"
                % (starter, ", ".join(sorted(RECOMMENDED)))
            )
        fetch_fn(item["repo"], model_path, require_new=False)
        fetched = True
    if not model_path.is_dir():
        raise ValueError(
            "first needs a checkpoint directory; %s is a file" % model_path
        )
    if not (0.0 < float(budget)):
        raise ValueError("budget must be positive")
    policy = policy or TeachPolicy()

    if session_factory is None:
        def session_factory(path, step_budget):
            from .session import Session

            return Session.open(
                path,
                dtype=dtype,
                device=device,
                quantize=quantize,
                budget=step_budget,
            )

    session = session_factory(str(model_path), EditBudget(max_total=budget))
    try:
        result = teach_one(
            session,
            intent.prompt,
            intent.answer,
            layers=layers,
            policy=policy,
            solver=solver,
            steps=steps,
            lr=lr,
            max_layers=max_layers,
        )
        payload = result.to_dict()
        final = None
        if result.status == "applied" and not no_save and out is not None:
            saved = save_fn(session, out)
            final = str(saved.get("output", out))
    finally:
        session = None
        gc.collect()

    verification = None
    if (
        verify_saved
        and final is not None
        and (check_fn is not None or real_runtime)
    ):
        checker = check_fn
        if checker is None:
            from .check import check_facts

            checker = check_facts
        try:
            verification = _as_dict(checker(
                final,
                [(intent.prompt, intent.answer)],
                dtype=dtype,
                device=device,
                quantize=quantize,
            ))
        except Exception as exc:
            verification = {
                "passed": False,
                "facts": [],
                "error": "%s: %s" % (type(exc).__name__, exc),
            }

    if fetched:
        prefix = "Downloaded %s and " % RECOMMENDED[starter]["name"]
    else:
        prefix = ""
    verified_ok = verification is None or bool(verification.get("passed"))
    status = result.status
    if result.status == "applied" and verification is not None and not verified_ok:
        status = "verification_failed"
    if status == "verification_failed":
        summary = (
            "%sapplied the fact and saved %s, but reloading those bytes did "
            "not reproduce the answer. Do not deploy this checkpoint before "
            "inspecting it." % (prefix, final)
        )
    elif result.status == "applied":
        if final:
            if verification is not None:
                summary = (
                    "%sapplied the fact and saved the checkpoint to %s. "
                    "Reloaded the saved bytes: %s."
                    % (prefix, final, "PASS" if verified_ok else "FAIL")
                )
            else:
                summary = "%sapplied the fact and saved the checkpoint to %s." % (
                    prefix, final,
                )
        else:
            summary = "%sapplied the fact; the edit is in memory only." % prefix
    elif result.status == "already_true":
        summary = "%sthe model already knew this fact; no edit was made." % prefix
    elif result.status == "rejected":
        summary = "%sthe local checks rejected the edit; the source is unchanged." % prefix
    else:
        summary = "%sresult: %s." % (prefix, result.status)
    return FirstResult(
        status=status,
        model=str(model_path),
        model_fetched=fetched,
        sentence=sentence,
        prompt=intent.prompt,
        answer=intent.answer,
        final=final,
        teach=payload,
        summary=summary,
        verification=verification,
    )
