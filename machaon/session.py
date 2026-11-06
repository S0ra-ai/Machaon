"""A loaded model plus the state an agent needs to edit it safely.

The session is the piece the MCP transport lacks: it holds a model, remembers
what it changed, and can put it back. Every operation is explicit — nothing is
applied until :meth:`apply` is called.

    session = Session.open("/models/qwen3b", dtype="float32")
    session.baseline(neutral_prompts, corpus_texts)
    proposal = session.propose(prompt, " Harvard", layer=26)
    session.apply()
    session.evaluate([prompt], [token_id])
    session.rollback()
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .budget import EditBudget
from .hygieia.metrics import kl_summary, perplexity_ratio, softmax
from .procedures.rome import RomeProposal, solve_edit
from .runtime.hf import HFModel

__all__ = ["Session", "SessionState"]


@dataclass
class SessionState:
    model: str
    parameters: int
    layers: int
    edited: List[str]
    budget_total: float
    budget_limit: Optional[float]
    has_baseline: bool


class Session:
    def __init__(self, runtime: HFModel, model_id: str, budget: Optional[EditBudget] = None):
        self.runtime = runtime
        self.model_id = model_id
        self.budget = budget or EditBudget(max_total=0.10)
        self._originals: Dict[str, np.ndarray] = {}
        self._pending: Optional[RomeProposal] = None
        self._baseline_logits: Optional[np.ndarray] = None
        self._baseline_prompts: List[str] = []
        self._baseline_nll: Optional[np.ndarray] = None
        self._baseline_corpus: List[str] = []

    @classmethod
    def open(cls, model_path: str, dtype: str = "float32",
             device: Optional[str] = None,
             budget: Optional[EditBudget] = None) -> "Session":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        dtypes = {
            "auto": None,
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if dtype not in dtypes:
            raise ValueError(
                "dtype must be one of %s" % ", ".join(sorted(dtypes))
            )
        kwargs = {}
        if dtypes[dtype] is not None:
            kwargs["torch_dtype"] = dtypes[dtype]
        if device == "auto":
            kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs).eval()
        if device is not None and device != "auto":
            model = model.to(device)
        return cls(HFModel(model, tokenizer), model_path, budget)

    # -- state ----------------------------------------------------------

    def state(self) -> SessionState:
        return SessionState(
            model=self.model_id,
            parameters=self.runtime.param_count(),
            layers=self.runtime.n_layers,
            edited=sorted(self._originals),
            budget_total=self.budget.state().total,
            budget_limit=self.budget.max_total,
            has_baseline=self._baseline_logits is not None,
        )

    # -- baseline -------------------------------------------------------

    def baseline(self, neutral_prompts: Sequence[str], corpus_texts: Sequence[str] = ()) -> Dict[str, Any]:
        """Capture reference distributions. Call this before the first edit."""
        if self._originals:
            raise RuntimeError("baseline must be captured before any edit is applied")
        self._baseline_prompts = list(neutral_prompts)
        self._baseline_corpus = list(corpus_texts)
        self._baseline_logits = (
            self.runtime.next_logits(self._baseline_prompts) if self._baseline_prompts else None
        )
        self._baseline_nll = (
            self.runtime.nll(self._baseline_corpus) if self._baseline_corpus else None
        )
        return {
            "prompts": len(self._baseline_prompts),
            "corpus": len(self._baseline_corpus),
        }

    # -- edit -----------------------------------------------------------

    def propose(self, prompt: str, target: str, layer: int, **kwargs) -> Dict[str, Any]:
        """Solve an edit without writing it. Replaces any pending proposal."""
        proposal = solve_edit(self.runtime, prompt, target, layer, **kwargs)
        self._pending = proposal
        return {
            "module": proposal.module,
            "layer": proposal.layer,
            "relative_delta": proposal.relative_delta,
            "target_probability": proposal.target_probability,
            "solver": proposal.solver,
            "steps": proposal.steps,
            "budget_after": self.budget.state().total + proposal.relative_delta,
        }

    def apply(self) -> Dict[str, Any]:
        """Write the pending proposal, charging the budget atomically."""
        if self._pending is None:
            raise RuntimeError("no pending proposal; call propose() first")
        proposal = self._pending
        name = proposal.module

        if name not in self._originals:
            original = self.runtime.read(name)
            self.budget.register(name, original)
            self._originals[name] = original
        # Check before writing, record after the write succeeds: a failed
        # write must not consume budget, and a charged edit must be visible.
        self.budget.check([(name, proposal.delta)])
        W = self.runtime.read(name)
        self.runtime.write(name, W + proposal.delta)
        try:
            self.budget.add([(name, proposal.delta)])
        except Exception:
            # The write already happened; put the bytes back before re-raising
            # so a charged-budget invariant never breaks.
            self.runtime.write(name, W)
            raise
        self._pending = None
        return {
            "module": name,
            "relative_delta": proposal.relative_delta,
            "budget_total": self.budget.state().total,
        }

    def rollback(self) -> Dict[str, Any]:
        """Restore every edited tensor to its original bytes."""
        restored = []
        failure: Optional[BaseException] = None
        for name, original in list(self._originals.items()):
            try:
                self.runtime.write(name, original)
            except BaseException as exc:
                failure = exc
                break
            restored.append(name)
            self._originals.pop(name, None)
        self._pending = None
        if not self._originals:
            self.budget = EditBudget(max_total=self.budget.max_total,
                                     max_per_tensor=self.budget.max_per_tensor)
        if failure is not None:
            raise failure
        return {"restored": sorted(restored)}

    # -- evaluation -----------------------------------------------------

    def token_id(self, text: str) -> int:
        """First token id of a string, for tools that speak text."""
        return int(self.runtime.tokenizer(text, add_special_tokens=False)["input_ids"][0])

    def evaluate(self, prompts: Sequence[str],
                 targets: Sequence[Any]) -> Dict[str, Any]:
        if not prompts:
            raise ValueError("no prompts to evaluate")
        logits = self.runtime.next_logits(prompts)
        predicted = np.asarray(logits).argmax(axis=-1)
        expected = np.asarray([
            self.token_id(t) if isinstance(t, str) else int(t) for t in targets
        ])
        if predicted.size != expected.size:
            raise ValueError("prompts and targets differ in length")

        metrics: Dict[str, Any] = {
            "accuracy": float((predicted == expected).mean()),
            "n": int(expected.size),
        }
        if self._baseline_logits is not None and self._baseline_prompts:
            current = self.runtime.next_logits(self._baseline_prompts)
            kl = kl_summary(softmax(self._baseline_logits), softmax(current))
            metrics.update({"kl_p50": kl["kl_p50"], "kl_p95": kl["kl_p95"],
                            "kl_max": kl["kl_max"]})
        if self._baseline_nll is not None and self._baseline_corpus:
            metrics["ppl_ratio"] = perplexity_ratio(
                self._baseline_nll, self.runtime.nll(self._baseline_corpus))
        return metrics
