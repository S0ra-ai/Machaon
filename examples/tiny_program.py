"""Run one complete weight program against a locally built tiny GPT-2.

This is an offline plumbing test: it downloads nothing, builds a tiny model
and tokenizer in a temporary directory, generates a program, teaches one fact,
reloads the saved bytes, and verifies the report again. It is intentionally
small so the full program lifecycle can run in a few seconds.

Requires the optional Hugging Face extras::

    pip install "machaon[hf]"
"""
from __future__ import annotations

import gc
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

WORDS = [
    "<unk>",
    "<|endoftext|>",
    "The",
    "answer",
    "is",
    "the",
    "capital",
    "of",
    "France",
    "Paris",
    "Two",
    "plus",
    "two",
    "equals",
    "Water",
    "freezes",
    "at",
    "a",
    "temperature",
    "Germany",
]


def _build_model(root: Path) -> Path:
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        GPT2Config,
        GPT2LMHeadModel,
        PreTrainedTokenizerFast,
    )

    model_dir = root / "model"
    model_dir.mkdir()
    vocab = {word: index for index, word in enumerate(WORDS)}
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        model_max_length=64,
    ).save_pretrained(str(model_dir))
    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=len(WORDS),
        n_positions=64,
        n_embd=64,
        n_layer=4,
        n_head=4,
    ))
    model.save_pretrained(str(model_dir), safe_serialization=True)
    return model_dir


def run_demo(root: Path) -> Dict[str, Any]:
    """Build the tiny checkpoint and execute the complete program lifecycle."""
    from machaon.program import (
        parse_program,
        render_program_text,
        run_program,
        verify_program_report,
    )

    model_dir = _build_model(root)
    text = render_program_text(
        model=str(model_dir),
        output=str(root / "out"),
        name="Offline tiny program",
        teach=["The answer is -> the"],
        protect=[
            "Two plus two equals",
            "Water freezes at a temperature of",
            "The capital of Germany is",
        ],
        budget=0.5,
        max_rel_norm=0.5,
        min_panel_agreement=0.0,
        layers=[1, 2],
        max_layers=2,
    )
    program_path = root / "tiny.machaon"
    program_path.write_text(text, encoding="utf-8")
    program = parse_program(
        text, source=str(program_path), base_dir=root
    )
    result = run_program(program, report_out=root / "program-report.json")
    checked = verify_program_report(
        root / "program-report.json",
        program_path=program_path,
        check_model=True,
    )
    gc.collect()
    return {
        "status": result.status,
        "accepted": result.accepted,
        "summary": result.summary,
        "report_path": result.report_path,
        "report_sha256": result.report_sha256,
        "verification": result.verification,
        "checked": checked,
    }


def main() -> int:
    try:
        with tempfile.TemporaryDirectory(prefix="machaon-tiny-program-") as tmp:
            payload = run_demo(Path(tmp))
    except ImportError as exc:
        print(
            "tiny_program needs the optional Hugging Face extras: "
            'pip install "machaon[hf]" (%s)' % exc,
            file=sys.stderr,
        )
        return 2
    print("Machaon · offline tiny program")
    print("status     : %s" % payload["status"].upper())
    print("summary    : %s" % payload["summary"])
    print("report SHA : %s" % payload["report_sha256"])
    checked = payload["checked"]["model_check"]
    print("model check: %s" % ("PASS" if checked["passed"] else "FAIL"))
    print("matches    : %s" % (
        "yes" if checked["matches_report"] else "no"
    ))
    return 0 if payload["accepted"] and checked["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
