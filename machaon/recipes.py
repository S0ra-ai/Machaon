"""One-call weight-space recipes.

The library primitives are deliberately explicit: choose a layer, capture a
key, solve a value, enforce a budget, evaluate, roll back.  A first-time user
should not have to know any of that.  :func:`teach_fact` runs the whole loop
from a prompt and an answer:

1. check whether the model already answers the prompt;
2. capture an unrelated baseline panel automatically;
3. try a spread of candidate layers, applying and rolling back each one;
4. re-solve the best layer and apply it;
5. measure the target, top-1 preservation and KL on the panel;
6. keep the edit only if every required local check passes, otherwise restore
   the original bytes.

This is not a claim of benchmark acceptance.  Metrics that need a benchmark
panel are listed in ``unmeasured``; the local checks only speak about the
target prompt and the declared panel.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .hygieia.metrics import softmax

__all__ = [
    "DEFAULT_BASELINE_PROMPTS",
    "TeachPolicy",
    "TeachResult",
    "candidate_layers",
    "teach",
    "teach_fact",
]


DEFAULT_BASELINE_PROMPTS = (
    "The capital of Germany is",
    "Water freezes at a temperature of",
    "Two plus two equals",
    "The opposite of hot is",
    "A dog is a kind of",
    "北京的首都是",
    "水在摄氏零度会",
    "太阳从东方升起，从西方",
)


def _finite(value: float, name: str, minimum: float = 0.0) -> float:
    value = float(value)
    if not math.isfinite(value) or value < minimum:
        raise ValueError("%s must be finite and >= %s" % (name, minimum))
    return value


@dataclass(frozen=True)
class TeachPolicy:
    """The local acceptance policy for one teaching edit.

    The defaults match the repository gates where those gates are measurable
    without a benchmark: relative norm 0.02, KL p95 0.05, perplexity ratio
    1.02.  ``min_panel_agreement`` permits one changed top-1 prediction in an
    eight-prompt panel; set it to 1.0 for a stricter run.
    """

    max_rel_norm: float = 0.02
    max_kl_p95: float = 0.05
    max_ppl_ratio: float = 1.02
    min_panel_agreement: float = 0.875
    require_panel: bool = True

    def __post_init__(self) -> None:
        if not (0.0 < _finite(self.max_rel_norm, "max_rel_norm") <= 0.5):
            raise ValueError("max_rel_norm must be in (0, 0.5]")
        if _finite(self.max_kl_p95, "max_kl_p95") <= 0.0:
            raise ValueError("max_kl_p95 must be positive")
        if _finite(self.max_ppl_ratio, "max_ppl_ratio") < 1.0:
            raise ValueError("max_ppl_ratio must be >= 1.0")
        if not (0.0 <= _finite(
            self.min_panel_agreement, "min_panel_agreement"
        ) <= 1.0):
            raise ValueError("min_panel_agreement must be in [0, 1]")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_rel_norm": float(self.max_rel_norm),
            "max_kl_p95": float(self.max_kl_p95),
            "max_ppl_ratio": float(self.max_ppl_ratio),
            "min_panel_agreement": float(self.min_panel_agreement),
            "require_panel": bool(self.require_panel),
        }


@dataclass
class TeachResult:
    """A JSON-safe record of one teaching attempt."""

    status: str
    prompt: str
    answer: str
    answer_text: str
    answer_token: int
    answer_token_text: str
    answer_token_count: int
    base_prediction: int
    base_prediction_text: str
    model: str
    n_layers: int
    layers: Tuple[int, ...]
    best_layer: Optional[int]
    tensor: Optional[str]
    relative_delta: Optional[float]
    budget_total: Optional[float]
    target_probability: Optional[float]
    base_accuracy: float
    candidate_accuracy: Optional[float]
    panel_size: int
    panel_agreement: Optional[float]
    metrics: Dict[str, Any] = field(default_factory=dict)
    base_metrics: Dict[str, Any] = field(default_factory=dict)
    checks: List[Dict[str, Any]] = field(default_factory=list)
    trials: List[Dict[str, Any]] = field(default_factory=list)
    ranking: List[Dict[str, Any]] = field(default_factory=list)
    policy: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    unmeasured: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)
    summary: str = ""
    applied: bool = False
    rolled_back: bool = False
    dry_run: bool = False
    elapsed_seconds: float = 0.0

    @property
    def accepted(self) -> bool:
        return self.status in ("applied", "already_true", "dry_run")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "accepted": bool(self.accepted),
            "prompt": self.prompt,
            "answer": self.answer,
            "answer_text": self.answer_text,
            "answer_token": int(self.answer_token),
            "answer_token_text": self.answer_token_text,
            "answer_token_count": int(self.answer_token_count),
            "base_prediction": int(self.base_prediction),
            "base_prediction_text": self.base_prediction_text,
            "model": self.model,
            "n_layers": int(self.n_layers),
            "layers": [int(layer) for layer in self.layers],
            "best_layer": None if self.best_layer is None else int(self.best_layer),
            "tensor": self.tensor,
            "relative_delta": self.relative_delta,
            "budget_total": self.budget_total,
            "target_probability": self.target_probability,
            "base_accuracy": float(self.base_accuracy),
            "candidate_accuracy": self.candidate_accuracy,
            "panel_size": int(self.panel_size),
            "panel_agreement": self.panel_agreement,
            "metrics": dict(self.metrics),
            "base_metrics": dict(self.base_metrics),
            "checks": list(self.checks),
            "trials": list(self.trials),
            "ranking": list(self.ranking),
            "policy": dict(self.policy),
            "notes": list(self.notes),
            "warnings": list(self.warnings),
            "unmeasured": list(self.unmeasured),
            "next_steps": list(self.next_steps),
            "summary": self.summary,
            "applied": bool(self.applied),
            "rolled_back": bool(self.rolled_back),
            "dry_run": bool(self.dry_run),
            "elapsed_seconds": float(self.elapsed_seconds),
        }


def candidate_layers(
    n_layers: int,
    count: int = 7,
    low: float = 0.55,
    high: float = 0.95,
) -> List[int]:
    """Return a depth-spread layer window.

    The default spans 55% to 95% of depth, which contains the measured edit
    sites for GPT-2, Qwen2.5 0.5B-7B and Llama-3.1-8B.  The caller can pass an
    explicit ``layers`` list to :func:`teach_fact` instead.
    """
    n_layers = int(n_layers)
    if n_layers < 1:
        raise ValueError("n_layers must be positive")
    count = int(count)
    if count < 1:
        raise ValueError("count must be positive")
    low = float(low)
    high = float(high)
    if not (0.0 <= low < high <= 1.0):
        raise ValueError("the layer window must satisfy 0 <= low < high <= 1")
    if n_layers == 1:
        return [0]
    count = min(count, n_layers)
    if count == 1:
        return [int(round(0.5 * (low + high) * (n_layers - 1)))]
    layers = []
    for index in range(count):
        position = low + (high - low) * index / (count - 1)
        layers.append(int(round(position * (n_layers - 1))))
    return sorted(set(layers))


def _state_field(state: Any, name: str, default: Any = None) -> Any:
    if isinstance(state, Mapping):
        return state.get(name, default)
    return getattr(state, name, default)


def _answer_variants(answer: str) -> List[str]:
    variants = [answer]
    stripped = answer.lstrip()
    if stripped != answer:
        variants.append(stripped)
    elif not answer.startswith(" "):
        variants.append(" " + answer)
    unique: List[str] = []
    for variant in variants:
        if variant not in unique:
            unique.append(variant)
    return unique


def _token_text(tokenizer: Any, token: int) -> str:
    decode = getattr(tokenizer, "decode", None)
    if callable(decode):
        try:
            text = decode([int(token)], skip_special_tokens=False)
        except Exception:
            text = None
        if isinstance(text, str) and text:
            return text if len(text) <= 64 else text[:61] + "..."
    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    if not callable(convert):
        return str(token)
    try:
        value = convert(int(token))
    except Exception:
        return str(token)
    if isinstance(value, (list, tuple)):
        value = value[0] if value else token
    text = str(value)
    return text if len(text) <= 64 else text[:61] + "..."


def _choose_answer(
    session: Any,
    prompt: str,
    answer: str,
) -> Tuple[str, int, str, int, Dict[str, Any]]:
    """Pick the tokenization a continuation is most likely to need.

    Answers are accepted with or without one leading space.  The variant with
    the larger base probability wins; ties prefer the spacing implied by the
    prompt, then the exact text the user wrote.  Nothing is written.
    """
    runtime = session.runtime
    tokenizer = getattr(runtime, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("teach_fact needs a tokenizer")
    logits = np.asarray(runtime.next_logits([prompt]))
    if logits.ndim != 2 or logits.shape[0] != 1:
        raise ValueError("next_logits must return one row for one prompt")
    probabilities = softmax(logits[0])
    expected_leading = not prompt.endswith((" ", "\n", "\t"))

    rows = []
    for variant in _answer_variants(answer):
        encoded = tokenizer(variant, add_special_tokens=False)["input_ids"]
        if not encoded:
            continue
        token = int(encoded[0])
        probability = float(probabilities[token]) if token < probabilities.size else 0.0
        leading = 1 if bool(variant.startswith(" ")) == expected_leading else 0
        exact = 1 if variant == answer else 0
        rows.append({
            "text": variant,
            "token": token,
            "probability": probability,
            "token_count": len(encoded),
            "score": (probability, leading, exact),
        })
    if not rows:
        raise ValueError("the answer produced no token")
    rows.sort(key=lambda row: row["score"], reverse=True)
    chosen = rows[0]
    base_prediction = int(np.argmax(probabilities))
    return (
        chosen["text"],
        int(chosen["token"]),
        _token_text(tokenizer, chosen["token"]),
        int(chosen["token_count"]),
        {
            "variants": [
                {
                    "text": row["text"],
                    "token": int(row["token"]),
                    "base_probability": float(row["probability"]),
                    "token_count": int(row["token_count"]),
                }
                for row in rows
            ],
            "base_prediction": base_prediction,
            "base_prediction_text": _token_text(tokenizer, base_prediction),
        },
    )


def _check(
    name: str,
    passed: bool,
    value: Any,
    requirement: str,
    *,
    required: bool = True,
    detail: str = "",
) -> Dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "value": value,
        "requirement": requirement,
        "required": bool(required),
        "detail": detail,
    }


def _percent(value: Optional[float]) -> str:
    if value is None:
        return "unknown"
    return "%.2f%%" % (100.0 * float(value))


def _plain_summary(
    status: str,
    prompt: str,
    answer_text: str,
    best_layer: Optional[int],
    tensor: Optional[str],
    relative_delta: Optional[float],
    panel_size: int,
    panel_agreement: Optional[float],
    reason: str,
) -> str:
    quoted = "%r -> %r" % (prompt, answer_text)
    if status == "already_true":
        return "The model already returns the target token %r for %r. " \
               "No edit was made." % (answer_text, prompt)
    if status == "dry_run":
        return (
            "Dry run: the best candidate is layer %s%s at %s of the tensor norm. "
            "No weights were written." % (
                "?" if best_layer is None else best_layer,
                "" if tensor is None else " (%s)" % tensor,
                _percent(relative_delta),
            )
        )
    if status == "rejected":
        return (
            "The edit for %s was reverted because %s. The original weights are "
            "restored." % (quoted, reason or "a local check failed")
        )
    agreement = (
        "unknown" if panel_agreement is None
        else "%d/%d" % (round(panel_agreement * panel_size), panel_size)
    )
    return (
        "The model did not return the target token %r for %r before the edit. "
        "Machaon changed %s at layer %s by %s of its norm. The target token now "
        "works and %s unrelated prompts kept their top-1 prediction." % (
            answer_text,
            prompt,
            tensor,
            best_layer,
            _percent(relative_delta),
            agreement,
        )
    )


def teach_fact(
    session: Any,
    prompt: str,
    answer: str,
    *,
    layers: Optional[Sequence[int]] = None,
    policy: Optional[TeachPolicy] = None,
    solver: str = "direction",
    steps: int = 250,
    lr: float = 0.02,
    max_layers: int = 7,
    layer_low: float = 0.55,
    layer_high: float = 0.95,
    preserve_prompts: Optional[Sequence[str]] = None,
    corpus_texts: Sequence[str] = (),
    max_evaluations: Optional[int] = None,
    notes: Sequence[str] = (),
    dry_run: bool = False,
) -> TeachResult:
    """Teach one prompt/answer pair, with automatic site search and rollback.

    ``session`` must be clean: no applied edits and no pending proposal.  The
    returned edit remains applied only when every required local check passes
    and ``dry_run`` is false.  A rejected edit is rolled back before return.
    """
    started = time.perf_counter()
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if not isinstance(answer, str) or answer == "":
        raise ValueError("answer must be a non-empty string")
    if solver not in ("direction", "gradient"):
        raise ValueError("solver must be 'direction' or 'gradient'")
    if int(steps) < 1:
        raise ValueError("steps must be positive")
    policy = policy or TeachPolicy()

    state = session.state()
    if _state_field(state, "edited") or _state_field(state, "has_pending"):
        raise RuntimeError(
            "teach_fact needs a clean session; roll back applied edits and "
            "clear the pending proposal first"
        )
    runtime = session.runtime
    n_layers = int(runtime.n_layers)
    model_id = str(getattr(session, "model_id", "<model>"))

    answer_text, answer_token, token_text, token_count, answer_evidence = _choose_answer(
        session, prompt, answer
    )
    base_prediction = int(answer_evidence["base_prediction"])
    base_prediction_text = str(answer_evidence["base_prediction_text"])
    warnings: List[str] = []
    if getattr(session, "quantize", None):
        warnings.append(
            "this session was loaded quantized; a quantized write can round the "
            "delta away. Use the Web console's densify control or a dense "
            "working set for the accepted edit"
        )
    if token_count > 1:
        warnings.append(
            "the answer is %d tokens; this edit pins only its first token %r"
            % (token_count, token_text)
        )
    if answer_text != answer and answer_text.strip() != answer.strip():
        warnings.append(
            "using %r instead of %r for the first token" % (answer_text, answer)
        )

    if layers is None:
        candidates = candidate_layers(
            n_layers, count=max_layers, low=layer_low, high=layer_high
        )
    else:
        candidates = [int(layer) for layer in layers]
    if not candidates:
        raise ValueError("at least one candidate layer is required")
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidate layers must be unique")
    if any(layer < 0 or layer >= n_layers for layer in candidates):
        raise ValueError(
            "candidate layers must be in [0, %d)" % n_layers
        )

    panel = [
        str(item) for item in (
            list(preserve_prompts)
            if preserve_prompts is not None
            else list(DEFAULT_BASELINE_PROMPTS)
        )
        if isinstance(item, str) and item.strip() and item != prompt
    ]
    panel = list(dict.fromkeys(panel))
    if not panel and policy.require_panel:
        raise ValueError("at least one unrelated preserve prompt is required")

    base_metrics = session.evaluate([prompt], [answer_token])
    base_accuracy = float(base_metrics.get("accuracy", 0.0))

    common = dict(
        prompt=prompt,
        answer=answer,
        answer_text=answer_text,
        answer_token=answer_token,
        answer_token_text=token_text,
        answer_token_count=token_count,
        base_prediction=base_prediction,
        base_prediction_text=base_prediction_text,
        model=model_id,
        n_layers=n_layers,
        layers=tuple(candidates),
        best_layer=None,
        tensor=None,
        relative_delta=None,
        budget_total=None,
        target_probability=None,
        base_accuracy=base_accuracy,
        candidate_accuracy=None,
        panel_size=len(panel),
        panel_agreement=None,
        base_metrics=dict(base_metrics),
        policy=policy.to_dict(),
        notes=list(notes),
        warnings=warnings,
        unmeasured=[
            "paraphrase_acc (full rewrite panel)",
            "unrelated_acc (full benchmark panel)",
            "mmlu_delta",
            "longform_contradictions",
        ],
    )
    if base_accuracy >= 1.0:
        result = TeachResult(
            status="already_true",
            summary=_plain_summary(
                "already_true", prompt, answer_text, None, None, None, len(panel), None, "",
            ),
            next_steps=["nothing to do; the model already returns the target token"],
            elapsed_seconds=time.perf_counter() - started,
            **common,
        )
        return result

    panel_base_logits = (
        np.asarray(runtime.next_logits(panel)).argmax(axis=-1) if panel else None
    )
    if not _state_field(session.state(), "has_baseline"):
        session.baseline(panel, list(corpus_texts))
    elif corpus_texts:
        warnings.append(
            "the session already had a baseline; the supplied corpus was not "
            "added, so perplexity is not measured"
        )

    search = session.search_rome_layers(
        prompt,
        answer_text,
        candidates,
        solver=solver,
        steps=int(steps),
        lr=float(lr),
        max_rel_norm=policy.max_rel_norm,
        preserve_prompts=panel,
        metric="accuracy",
        max_evaluations=max_evaluations,
    )
    trials = list(search.get("trials", []))
    ranking = list(search.get("ranking", []))
    hits = [
        trial for trial in trials
        if trial.get("status") == "ok"
        and float(trial.get("metrics", {}).get("accuracy", 0.0)) >= 1.0
    ]

    if not hits:
        reason = "no candidate layer reached the target token"
        result = TeachResult(
            status="rejected",
            trials=trials,
            ranking=ranking,
            checks=[_check("target", False, 0.0, "accuracy == 1.0")],
            summary=_plain_summary(
                "rejected", prompt, answer_text, None, None, None,
                len(panel), None, reason,
            ),
            next_steps=[
                "widen --layers, or try --solver gradient",
                "check that the answer token is the continuation the prompt expects",
            ],
            rolled_back=False,
            elapsed_seconds=time.perf_counter() - started,
            **common,
        )
        return result

    def trial_score(trial: Mapping[str, Any]) -> Tuple[float, float, int]:
        metrics = trial.get("metrics", {})
        kl = metrics.get("kl_p95")
        kl_value = float(kl) if kl is not None else math.inf
        rel = float(metrics.get("relative_delta", math.inf))
        return (kl_value, rel, int(trial["layer"]))

    winner = sorted(hits, key=trial_score)[0]
    best_layer = int(winner["layer"])
    if dry_run:
        dry_common = dict(common)
        dry_common["best_layer"] = best_layer
        dry_common["relative_delta"] = float(
            winner["metrics"].get("relative_delta", 0.0)
        )
        result = TeachResult(
            status="dry_run",
            trials=trials,
            ranking=ranking,
            checks=[
                _check(
                    "target", True,
                    float(winner["metrics"].get("accuracy", 0.0)),
                    "accuracy == 1.0",
                    detail="measured during the search; the trial was rolled back",
                )
            ],
            summary=_plain_summary(
                "dry_run", prompt, answer_text, best_layer, None,
                float(winner["metrics"].get("relative_delta", 0.0)),
                len(panel), None, "",
            ),
            next_steps=["rerun without --dry-run to apply the edit"],
            dry_run=True,
            elapsed_seconds=time.perf_counter() - started,
            **dry_common,
        )
        return result

    try:
        proposal = session.propose(
            prompt,
            answer_text,
            best_layer,
            solver=solver,
            steps=int(steps),
            lr=float(lr),
            max_rel_norm=policy.max_rel_norm,
            preserve_prompts=panel,
        )
        applied = session.apply()
        candidate_metrics = session.evaluate([prompt], [answer_token])
        panel_agreement = None
        if panel and panel_base_logits is not None:
            panel_after = np.asarray(runtime.next_logits(panel)).argmax(axis=-1)
            panel_agreement = float((panel_after == panel_base_logits).mean())
    except Exception as exc:
        try:
            session.rollback()
        except Exception as rollback_exc:
            raise RuntimeError(
                "teaching the edit failed (%s) and rollback also failed (%s)"
                % (exc, rollback_exc)
            ) from exc
        raise

    checks = [
        _check(
            "target",
            float(candidate_metrics.get("accuracy", 0.0)) >= 1.0,
            float(candidate_metrics.get("accuracy", 0.0)),
            "accuracy == 1.0",
        ),
        _check(
            "budget",
            float(applied.get("relative_delta", 0.0)) <= policy.max_rel_norm + 1e-12,
            float(applied.get("relative_delta", 0.0)),
            "relative_delta <= %.6f" % policy.max_rel_norm,
        ),
    ]
    if panel:
        checks.append(_check(
            "panel_top1",
            panel_agreement is not None
            and panel_agreement >= policy.min_panel_agreement,
            panel_agreement,
            ">= %.3f of the panel" % policy.min_panel_agreement,
            detail="top-1 prediction preserved on unrelated prompts",
        ))
    if "kl_p95" in candidate_metrics:
        checks.append(_check(
            "kl_p95",
            float(candidate_metrics["kl_p95"]) <= policy.max_kl_p95,
            float(candidate_metrics["kl_p95"]),
            "<= %.4f" % policy.max_kl_p95,
        ))
    else:
        warnings.append("KL was not measured; capture a baseline first")
    if "ppl_ratio" in candidate_metrics:
        checks.append(_check(
            "ppl_ratio",
            float(candidate_metrics["ppl_ratio"]) <= policy.max_ppl_ratio,
            float(candidate_metrics["ppl_ratio"]),
            "<= %.4f" % policy.max_ppl_ratio,
            required=False,
            detail="optional; measured only when a corpus was supplied",
        ))

    failed = [
        check for check in checks
        if check["required"] and not check["passed"]
    ]
    common.update(
        best_layer=best_layer,
        tensor=proposal.get("module"),
        relative_delta=float(applied.get("relative_delta", 0.0)),
        budget_total=float(applied.get("budget_total", 0.0)),
        target_probability=float(proposal.get("target_probability", 0.0)),
        candidate_accuracy=float(candidate_metrics.get("accuracy", 0.0)),
        panel_agreement=panel_agreement,
        metrics=dict(candidate_metrics),
        checks=checks,
        trials=trials,
        ranking=ranking,
    )

    if failed:
        reason = "%s failed (%s)" % (
            failed[0]["name"], failed[0]["requirement"],
        )
        rolled_back = False
        try:
            session.rollback()
            rolled_back = True
        except Exception as exc:
            raise RuntimeError(
                "teach_fact rejected the edit but rollback failed: %s" % exc
            ) from exc
        result = TeachResult(
            status="rejected",
            summary=_plain_summary(
                "rejected", prompt, answer_text, best_layer, proposal.get("module"),
                float(applied.get("relative_delta", 0.0)), len(panel),
                panel_agreement, reason,
            ),
            next_steps=[
                "try another layer window, or --solver gradient",
                "lower the edit budget only if the target is still reached",
            ],
            rolled_back=rolled_back,
            elapsed_seconds=time.perf_counter() - started,
            **common,
        )
        return result

    result = TeachResult(
        status="applied",
        summary=_plain_summary(
            "applied", prompt, answer_text, best_layer, proposal.get("module"),
            float(applied.get("relative_delta", 0.0)), len(panel),
            panel_agreement, "",
        ),
        next_steps=[
            "save the checkpoint, or keep this session for further manual edits",
            "start a new session for another teaching edit",
            "run a full gate report before claiming benchmark acceptance",
        ],
        applied=True,
        elapsed_seconds=time.perf_counter() - started,
        **common,
    )
    return result


def teach(
    session: Any,
    sentence: str,
    answer: Optional[str] = None,
    **kwargs: Any,
) -> TeachResult:
    """The friendly entry point: a sentence or a prompt/answer pair.

    ``teach(session, "The Eiffel Tower is in the city of -> Rome")`` compiles
    the sentence with :func:`machaon.intent.parse_intent`.  Passing ``answer``
    explicitly skips the compiler and uses the prompt as given.
    """
    if answer is None:
        from .intent import parse_intent

        intent = parse_intent(sentence)
        return teach_fact(session, intent.prompt, intent.answer, **kwargs)
    return teach_fact(session, sentence, answer, **kwargs)
