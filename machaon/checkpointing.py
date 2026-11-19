"""Write a live edit session to a loadable checkpoint directory."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict

__all__ = ["save_session", "checkpoint_problem"]


def checkpoint_problem(path: str | Path) -> str:
    """Describe why a checkpoint directory cannot be opened, if it cannot.

    Interrupted downloads are the most common first-run failure: a user sees a
    traceback from the tokenizer instead of "your download is incomplete".
    Returns an empty string when nothing obvious is wrong.
    """
    source = Path(path).expanduser()
    if not source.is_dir():
        return ""
    partial = sorted(source.glob("*.part"))
    if partial:
        return (
            "the checkpoint still contains an unfinished download (%s); "
            "rerun machaon fetch to resume it" % partial[0].name
        )
    if not (source / "config.json").exists():
        return "the checkpoint has no config.json; it is not a Hugging Face checkpoint"
    weights = (
        any(source.glob("*.safetensors"))
        or any(source.glob("*.bin"))
        or (source / "model.safetensors.index.json").exists()
    )
    if not weights:
        return "the checkpoint has no weight files; the download may be incomplete"
    tokenizer = (
        (source / "tokenizer.json").exists()
        or (source / "tokenizer.model").exists()
        or ((source / "vocab.json").exists() and (source / "merges.txt").exists())
        or (source / "spiece.model").exists()
    )
    if not tokenizer:
        return "the checkpoint has no tokenizer files; the download may be incomplete"
    return ""


def save_session(session: Any, out_dir: str | Path,
                 require_new: bool = True) -> Dict[str, Any]:
    """Persist every applied delta through the safetensors adapter.

    The runtime speaks canonical ``(d_out, d_in)`` orientation; the writer must
    speak the checkpoint's stored layout, so this is the one place that applies
    ``runtime.stored_values``.  A failed write removes the partial directory.
    """
    target = Path(out_dir).expanduser()
    if require_new and target.exists():
        raise FileExistsError(
            "output already exists: %s (choose another path or remove it)" % target
        )
    changes = session.changes()
    if not changes:
        raise RuntimeError("the session has no applied edits to save")
    from .adapters import Checkpoint

    checkpoint = Checkpoint(session.model_id)
    stored = getattr(session.runtime, "stored_values", None)
    try:
        if len(changes) == 1:
            name = next(iter(changes))
            values = session.runtime.read(name)
            if callable(stored):
                values = stored(name, values)
            report = checkpoint.write_tensor(name, values, target)
        else:
            if callable(stored):
                changes = {
                    name: stored(name, delta) for name, delta in changes.items()
                }
            report = checkpoint.write_with(changes, target)
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise
    return {"output": str(target), "checkpoint": report}
