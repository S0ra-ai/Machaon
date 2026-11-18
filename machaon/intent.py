"""Compile a plain-language request into an explicit edit contract.

This is a deliberately small language, not a language model.  It accepts the
forms that determine one prompt and one answer without guessing, shows the
parse, and refuses everything else with examples.  The result is the same
``prompt`` / ``answer`` pair that :func:`machaon.recipes.teach_fact` consumes,
so a one-sentence request never bypasses the budget, transaction or evidence
layer.

Supported forms::

    The Eiffel Tower is in the city of -> Rome
    prompt: The Eiffel Tower is in the city of
    answer:  Rome
    when asked "The Eiffel Tower is in the city of", answer " Rome"
    teach the model to answer " Rome" when asked "The Eiffel Tower is in the city of"
    teach the model that the Eiffel Tower is in Paris
    教会模型：北京的首都是 → 北京
    当问到「北京的首都是」时，回答「北京」

The arrow and labelled forms are exact.  The "teach the model that ..." form
is a convenience that splits at the first copula and is marked medium
confidence, because the split point is an assumption.
"""
from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = ["EXAMPLES", "IntentError", "TeachIntent", "parse_intent"]

EXAMPLES = (
    "The Eiffel Tower is in the city of -> Rome",
    'when asked "The Eiffel Tower is in the city of", answer " Rome"',
    "prompt: The Eiffel Tower is in the city of\nanswer:  Rome",
    "教会模型：北京的首都是 → 北京",
    "当问到「北京的首都是」时，回答「北京」",
)

_ARROWS = ("⟶", "⇒", "⇨", "->", "=>", "→")
_QUOTE_PAIRS = {
    '"': '"',
    "'": "'",
    "“": "”",
    "‘": "’",
    "「": "」",
    "『": "』",
    "《": "》",
}
_TERMINAL = ".。!?！？"
_LABEL_KEYS = {
    "prompt": (
        "prompt", "question", "input", "key",
        "提示词", "提示", "问题", "输入",
    ),
    "answer": (
        "answer", "target", "output", "completion",
        "回答", "答案", "输出",
    ),
}


class IntentError(ValueError):
    """The text does not determine one prompt and one answer."""

    def __init__(self, message: str, examples: Optional[Sequence[str]] = None) -> None:
        self.examples = tuple(examples or EXAMPLES)
        rendered = "\n".join("  " + example for example in self.examples)
        super().__init__("%s\nTry one of:\n%s" % (message, rendered))


@dataclass(frozen=True)
class TeachIntent:
    """One reviewable teaching request."""

    prompt: str
    answer: str
    source: str = "arrow"
    confidence: str = "high"
    notes: Tuple[str, ...] = ()
    kind: str = "teach"

    def __post_init__(self) -> None:
        if self.kind != "teach":
            raise ValueError("this release compiles teach intents only")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if not isinstance(self.answer, str) or self.answer == "":
            raise ValueError("answer must be a non-empty string")
        if self.confidence not in ("high", "medium"):
            raise ValueError("confidence must be high or medium")

    def to_dict(self) -> Dict[str, object]:
        return {
            "kind": self.kind,
            "prompt": self.prompt,
            "answer": self.answer,
            "source": self.source,
            "confidence": self.confidence,
            "notes": list(self.notes),
        }

    def to_command(self, model: str = "<checkpoint>") -> str:
        """The exact CLI command that runs this parse."""
        return "machaon teach --model %s --prompt %s --answer %s" % (
            shlex.quote(model),
            shlex.quote(self.prompt),
            shlex.quote(self.answer),
        )


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] in _QUOTE_PAIRS:
        if value[-1] == _QUOTE_PAIRS[value[0]]:
            return value[1:-1]
    return value


def _clean_side(value: str, *, strip_terminal: bool = False) -> str:
    value = value.strip()
    if strip_terminal and value and value[-1] in _TERMINAL:
        if _unquote(value) is value:
            value = value[:-1].rstrip()
    return _unquote(value)


def _try_labeled(text: str) -> Optional[Tuple[str, str]]:
    """Parse a prompt/answer block, preserving one space after the colon.

    A value written as ``answer:  Rome`` keeps one leading space, which is the
    token boundary most tokenizers expect after a prompt that does not end in
    whitespace.  ``answer: Rome`` keeps none.  Quotes preserve the value
    exactly.
    """
    values: Dict[str, List[str]] = {"prompt": [], "answer": []}
    current: Optional[str] = None
    saw_label = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        match = re.match(r"^([^:：]{1,20})[:：]([ \t]?)(.*)$", line)
        if match:
            label = match.group(1).strip().lower()
            kind = None
            for candidate, names in _LABEL_KEYS.items():
                if label in names:
                    kind = candidate
                    break
            if kind is not None:
                saw_label = True
                current = kind
                values[kind].append(match.group(3))
                continue
        if current is None:
            return None
        values[current].append(line)
    if not saw_label:
        return None
    prompt = _unquote("\n".join(values["prompt"]).strip())
    answer = _unquote(" ".join(values["answer"]))
    if not prompt or not answer.strip():
        raise IntentError("a labelled request needs both a prompt and an answer")
    return prompt, answer


def _split_arrow(text: str) -> Optional[Tuple[str, str, str]]:
    positions = []
    for arrow in _ARROWS:
        index = text.find(arrow)
        if index >= 0:
            positions.append((index, -len(arrow), arrow))
    if not positions:
        return None
    index, _, arrow = min(positions)
    left = text[:index]
    right = text[index + len(arrow):]
    return left, right, arrow


def _strip_teach_prefix(value: str) -> str:
    value = value.strip()
    prefixes = (
        "teach the model:",
        "teach:",
        "教会模型：",
        "教会模型:",
        "教模型：",
        "教模型:",
        "让模型：",
        "让模型:",
    )
    lowered = value.lower()
    for prefix in prefixes:
        if lowered.startswith(prefix.lower()):
            return value[len(prefix):].strip()
    return value


_PHRASE_PATTERNS = (
    (
        "phrase",
        re.compile(
            r"^teach\s+(?:the\s+)?model\s+to\s+(?:answer|say|reply)(?:\s+with)?\s+"
            r"(?P<a>.+?)\s+when\s+asked\s+(?P<p>.+)$",
            re.IGNORECASE,
        ),
    ),
    (
        "phrase",
        re.compile(
            r"^(?:when|if)\s+(?:the\s+model\s+is\s+)?asked\s+(?P<p>.+?)\s*[,;]\s*"
            r"(?:the\s+model\s+should\s+|it\s+should\s+|please\s+)?"
            r"(?:answer|say|reply|respond)(?:\s+with)?\s+(?P<a>.+)$",
            re.IGNORECASE,
        ),
    ),
    (
        "phrase",
        re.compile(
            r"^(?:when|if)\s+asked\s+(?P<p>.+?)\s*[,;]?\s+"
            r"(?:answer|say|reply|respond)(?:\s+with)?\s+(?P<a>.+)$",
            re.IGNORECASE,
        ),
    ),
    (
        "phrase",
        re.compile(
            r"^(?:当|如果)?(?:有人|用户|我)?(?:问到|问起|问)[：:]?\s*"
            r"(?P<p>.+?)(?:的时候|时)?\s*[，,]\s*(?:就|请|让(?:它|他))?"
            r"(?:回答|答|说)(?:成|为)?[：:]?\s*(?P<a>.+)$"
        ),
    ),
    (
        "phrase",
        re.compile(
            r"^(?:教(?:会)?|让)模型[：:]?\s*(?P<p>.+?)\s*[，,]\s*"
            r"(?:让(?:它|他)?)?(?:回答|答|说)[：:]?\s*(?P<a>.+)$"
        ),
    ),
    (
        "phrase",
        re.compile(
            r"^(?:让|教(?:会)?)模型在\s*(?P<p>.+?)\s*(?:的时候|时)\s*"
            r"(?:回答|答|说)[：:]?\s*(?P<a>.+)$"
        ),
    ),
)

_SENTENCE_PATTERNS = (
    (
        "sentence",
        re.compile(
            r"^(?:teach|make)\s+(?:the\s+)?model\s+"
            r"(?:that\s+|know\s+that\s+)?(?P<body>.+)$",
            re.IGNORECASE,
        ),
    ),
    (
        "sentence",
        re.compile(r"^教(?:会)?模型\s*(?P<body>.+)$"),
    ),
)

_COPULA = re.compile(r"\b(is|are|was|were|means|equals)\b", re.IGNORECASE)


def _split_sentence(body: str) -> Optional[Tuple[str, str, str]]:
    match = _COPULA.search(body)
    if match:
        return body[:match.end()], body[match.end():], match.group(1)
    index = body.find("是")
    if index >= 0:
        return body[: index + 1], body[index + 1:], "是"
    return None


def _clean_sentence_answer(value: str) -> str:
    """Keep the leading space created by the copula split."""
    if value and value[-1] in _TERMINAL and _unquote(value) is value:
        value = value[:-1]
    return _unquote(value)


def _is_forget_command(text: str) -> bool:
    return bool(re.match(
        r"^\s*(?:forget\b|make\s+.*\bforget\b|让模型忘记|让模型忘掉|删除模型)",
        text,
        re.IGNORECASE,
    ))


def parse_intent(text: str) -> TeachIntent:
    """Parse one teaching request or raise :class:`IntentError`.

    The function never contacts a model and never guesses a layer.  It only
    decides what must change; choosing where and how is the recipe's job.
    """
    if not isinstance(text, str) or not text.strip():
        raise IntentError("the request is empty")
    if len(text) > 8000:
        raise IntentError("the request is too long; keep it under 8000 characters")
    raw = text.strip()

    labelled = _try_labeled(raw)
    if labelled is not None:
        prompt, answer = labelled
        return TeachIntent(prompt=prompt, answer=answer, source="labels")

    arrow = _split_arrow(raw)
    if arrow is not None:
        left, right, _ = arrow
        prompt = _clean_side(_strip_teach_prefix(left))
        answer = _clean_side(right, strip_terminal=True)
        if not prompt or not answer:
            raise IntentError("the arrow form needs text on both sides")
        return TeachIntent(prompt=prompt, answer=answer, source="arrow")

    flat = " ".join(line.strip() for line in raw.splitlines() if line.strip())
    for source, pattern in _PHRASE_PATTERNS:
        match = pattern.match(flat)
        if not match:
            continue
        prompt = _clean_side(match.group("p"))
        answer = _clean_side(match.group("a"), strip_terminal=True)
        if not prompt or not answer:
            raise IntentError("the request needs both a prompt and an answer")
        return TeachIntent(prompt=prompt, answer=answer, source=source)

    for source, pattern in _SENTENCE_PATTERNS:
        match = pattern.match(flat)
        if not match:
            continue
        body = match.group("body").strip()
        split = _split_sentence(body)
        if split is None:
            raise IntentError(
                "could not find a clause to split into prompt and answer; "
                "use an arrow or a labelled block",
                examples=EXAMPLES,
            )
        prompt, answer, marker = split
        answer = _clean_sentence_answer(answer)
        if not prompt or not answer:
            raise IntentError("the sentence split produced an empty side")
        return TeachIntent(
            prompt=prompt,
            answer=answer,
            source=source,
            confidence="medium",
            notes=(
                "split at %r; use an arrow for an exact prompt/answer pair" % marker,
            ),
        )

    if _is_forget_command(raw):
        raise IntentError(
            "this release compiles teach requests only; to change a behaviour, "
            "give the prompt and the new answer",
            examples=EXAMPLES,
        )
    raise IntentError("could not read this as a teaching request")
