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
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .action import ActionResult, ActionSpec, execute_action
from .batch import BatchSolution, solve_batch
from .budget import EditBudget
from .headroom import DecodingConfig, score_generation_texts
from .hygieia.metrics import kl_summary, perplexity_ratio, softmax
from .mechanism import MechanismProtocol, generation_mechanism
from .planning import search_layers as _search_layers
from .procedures.rome import RomeProposal, solve_edit
from .runtime.hf import HFModel
from .schema import Target
from .wsp import WSPProtocol, assess_wsp, random_controls

__all__ = ["Session", "SessionState", "BatchProposal", "ActionProposal"]


@dataclass
class BatchProposal:
    """A solved multi-key edit that has not been written yet."""

    module: str
    layer: int
    delta: np.ndarray
    solution: BatchSolution


@dataclass
class ActionProposal:
    """A validated, isolated Python action that has not been written yet."""

    module: str
    layer: int
    result: ActionResult


@dataclass
class SessionState:
    model: str
    parameters: int
    layers: int
    edited: List[str]
    budget_total: float
    budget_limit: Optional[float]
    has_baseline: bool
    device: str = "cpu"
    quantize: Optional[str] = None
    has_pending: bool = False


class Session:
    def __init__(self, runtime: HFModel, model_id: str,
                 budget: Optional[EditBudget] = None,
                 quantize: Optional[str] = None):
        self.runtime = runtime
        self.model_id = model_id
        self.budget = budget or EditBudget(max_total=0.10)
        self.quantize = quantize
        self._originals: Dict[str, np.ndarray] = {}
        self._pending: Optional[RomeProposal] = None
        self._pending_batch: Optional[BatchProposal] = None
        self._pending_action: Optional[ActionProposal] = None
        self._baseline_logits: Optional[np.ndarray] = None
        self._baseline_prompts: List[str] = []
        self._baseline_nll: Optional[np.ndarray] = None
        self._baseline_corpus: List[str] = []

    @classmethod
    def open(cls, model_path: str, dtype: str = "float32",
             device: Optional[str] = None,
             budget: Optional[EditBudget] = None,
             quantize: Optional[str] = None) -> "Session":
        """Load a checkpoint, optionally with a quantized working set.

        ``quantize`` accepts ``"int4"`` or ``"int8"`` through Transformers'
        ``QuantoConfig``.  Quantized loading uses ``device_map`` and skips a
        second ``.to(device)`` pass, which keeps a 27B checkpoint within a
        laptop's unified-memory envelope.  The edited layer can then be
        densified with ``runtime.densify_from(layer)`` before a gradient solve.
        """
        import torch
        from transformers import AutoTokenizer

        from .runtime.hf import load_pretrained

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
        if quantize not in (None, "int4", "int8"):
            raise ValueError("quantize must be one of int4, int8 or None")
        if quantize is not None and dtype != "auto":
            raise ValueError("quantize requires dtype='auto'")
        kwargs = {}
        if dtypes[dtype] is not None and quantize is None:
            kwargs["torch_dtype"] = dtypes[dtype]
        if quantize is not None:
            try:
                from transformers import QuantoConfig
            except ImportError as exc:
                raise ImportError(
                    "quantized Session.open needs optimum-quanto: "
                    'pip install "machaon[hf]" optimum-quanto'
                ) from exc
            kwargs["quantization_config"] = QuantoConfig(weights=quantize)
            kwargs["device_map"] = "auto" if device == "auto" else (device or "cpu")
        elif device == "auto":
            kwargs["device_map"] = "auto"
        model = load_pretrained(model_path, **kwargs).eval()
        if quantize is None and device is not None and device != "auto":
            model = model.to(device)
        return cls(HFModel(model, tokenizer), model_path, budget, quantize=quantize)

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
            device=str(self.runtime.device),
            quantize=self.quantize,
            has_pending=(
                self._pending is not None
                or self._pending_batch is not None
                or self._pending_action is not None
            ),
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
        self._pending_batch = None
        self._pending_action = None
        return {
            "module": proposal.module,
            "layer": proposal.layer,
            "relative_delta": proposal.relative_delta,
            "target_probability": proposal.target_probability,
            "solver": proposal.solver,
            "steps": proposal.steps,
            "budget_after": self.budget.state().total + proposal.relative_delta,
        }

    def propose_batch(
        self,
        prompts: Sequence[str],
        values: Any,
        layer: int,
        *,
        module_suffix: str = "mlp.down_proj",
        preserve_prompts: Sequence[str] = (),
        max_rel_norm: Optional[float] = None,
        clamp: bool = True,
        ridge: float = 0.0,
    ) -> Dict[str, Any]:
        """Solve many prompt/activation/value rows into one pending delta."""
        if self.runtime.tokenizer is None:
            raise RuntimeError("propose_batch needs a tokenizer")
        prompts = list(prompts)
        if not prompts:
            raise ValueError("propose_batch needs at least one prompt")
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] != len(prompts):
            raise ValueError("values must be (len(prompts), d_out)")

        weight_name = self.runtime.edit_target(layer, module_suffix)
        module_name = weight_name[: -len(".weight")]
        keys = np.stack([
            self.runtime.capture_module_input(
                module_name, self.runtime.encode([prompt])[0]
            )[0]
            for prompt in prompts
        ])
        preserve = None
        if preserve_prompts:
            preserve = np.stack([
                self.runtime.capture_module_input(
                    module_name, self.runtime.encode([prompt])[0]
                )[0]
                for prompt in preserve_prompts
            ])
        W = self.runtime.read(weight_name)
        solution = solve_batch(
            W, keys, values,
            preserve=preserve,
            max_rel_norm=max_rel_norm,
            clamp=clamp,
            ridge=ridge,
        )
        self._pending = None
        self._pending_batch = BatchProposal(
            module=weight_name, layer=int(layer), delta=solution.delta,
            solution=solution,
        )
        self._pending_action = None
        return {
            "module": weight_name,
            "layer": int(layer),
            "relative_delta": solution.relative_norm,
            "target_error": solution.target_error,
            "per_key_error": list(solution.per_key_error),
            "exact": solution.exact,
            "clamped": solution.clamped,
            "rank": solution.rank,
            "budget_after": self.budget.state().total + solution.relative_norm,
        }

    def propose_action(
        self,
        source: str,
        intent: str,
        layer: int,
        *,
        module_suffix: str = "mlp.down_proj",
        keys: Any = None,
        values: Any = None,
        preserve: Any = None,
        preserve_prompts: Sequence[str] = (),
        max_rel_norm: float = 0.02,
        timeout_seconds: float = 10.0,
        memory_mb: int = 1024,
    ) -> Dict[str, Any]:
        """Execute one constrained Python action into a pending delta.

        The action is a pure ``def edit(W): ...`` function over one selected
        tensor.  It runs in a separate process under the AST whitelist in
        :mod:`machaon.action`; nothing is written until :meth:`apply`.
        """
        weight_name = self.runtime.edit_target(layer, module_suffix)
        if preserve_prompts:
            if self.runtime.tokenizer is None:
                raise RuntimeError("propose_action preserve_prompts needs a tokenizer")
            module_name = weight_name[: -len(".weight")]
            preserve = np.stack([
                self.runtime.capture_module_input(
                    module_name, self.runtime.encode([prompt])[0]
                )[0]
                for prompt in preserve_prompts
            ])
        spec = ActionSpec(
            source=source,
            target=Target(weight_name, int(layer)),
            intent=intent,
            max_rel_norm=max_rel_norm,
            timeout_seconds=timeout_seconds,
            memory_mb=memory_mb,
        )
        result = execute_action(
            spec,
            self.runtime.read(weight_name),
            keys=keys,
            values=values,
            preserve=preserve,
        )
        self._pending = None
        self._pending_batch = None
        self._pending_action = ActionProposal(
            module=weight_name,
            layer=int(layer),
            result=result,
        )
        return {
            "module": weight_name,
            "layer": int(layer),
            "relative_delta": result.relative_norm,
            "source_sha256": result.source_sha256,
            "elapsed_ms": result.elapsed_ms,
            "protected": result.protected,
            "budget_after": self.budget.state().total + result.relative_norm,
        }

    def _apply_delta(self, name: str, delta: np.ndarray,
                     relative_delta: float,
                     extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if name not in self._originals:
            original = self.runtime.read(name)
            self.budget.register(name, original)
            self._originals[name] = original
        # Check before writing, record after the write succeeds: a failed
        # write must not consume budget, and a charged edit must be visible.
        self.budget.check([(name, delta)])
        W = self.runtime.read(name)
        self.runtime.write(name, W + delta)
        try:
            self.budget.add([(name, delta)])
        except Exception:
            # The write already happened; put the bytes back before re-raising
            # so a charged-budget invariant never breaks.
            self.runtime.write(name, W)
            raise
        self._pending = None
        self._pending_batch = None
        self._pending_action = None
        result = {
            "module": name,
            "relative_delta": relative_delta,
            "budget_total": self.budget.state().total,
        }
        if extra:
            result.update(extra)
        return result

    def apply(self) -> Dict[str, Any]:
        """Write the pending proposal, charging the budget atomically."""
        if self._pending_action is not None:
            proposal = self._pending_action
            result = proposal.result
            return self._apply_delta(
                proposal.module,
                result.delta,
                result.relative_norm,
                {
                    "source_sha256": result.source_sha256,
                    "intent": result.intent,
                    "protected": result.protected,
                },
            )
        if self._pending_batch is not None:
            proposal = self._pending_batch
            return self._apply_delta(
                proposal.module,
                proposal.delta,
                proposal.solution.relative_norm,
                {
                    "target_error": proposal.solution.target_error,
                    "per_key_error": list(proposal.solution.per_key_error),
                    "exact": proposal.solution.exact,
                    "clamped": proposal.solution.clamped,
                    "rank": proposal.solution.rank,
                },
            )
        if self._pending is None:
            raise RuntimeError("no pending proposal; call propose() first")
        proposal = self._pending
        return self._apply_delta(
            proposal.module, proposal.delta, proposal.relative_delta
        )

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
        self._pending_batch = None
        self._pending_action = None
        if not self._originals:
            self.budget = EditBudget(max_total=self.budget.max_total,
                                     max_per_tensor=self.budget.max_per_tensor)
        if failure is not None:
            raise failure
        return {"restored": sorted(restored)}

    def changes(self) -> Dict[str, np.ndarray]:
        """Return the current deltas for every applied tensor.

        This is the bridge from an in-memory session to a checkpoint writer:
        the returned arrays are ``current - original`` in the same canonical
        orientation as :meth:`HFModel.read`.  It does not clear the originals;
        rollback remains available after a save.
        """
        return {
            name: self.runtime.read(name) - original
            for name, original in self._originals.items()
        }

    def export_openwsp(
        self,
        path: str,
        *,
        base_model: Optional[str] = None,
        revision: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        form: str = "sparse",
        rank: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Export the applied session deltas as an OpenWSP v1 edit.

        The export is sparse and bf16-rounded, matching the external format.
        ``metadata`` is stored under the manifest's ``machaon`` object and is
        therefore committed by ``edit_sha256``; this is the hook for binding a
        WSP evidence address to the edit.
        """
        if not self._originals:
            raise RuntimeError("no applied edits to export")
        from .interop import openwsp

        originals = {name: value.copy() for name, value in self._originals.items()}
        edited = {name: self.runtime.read(name) for name in originals}
        return openwsp.export_edit(
            originals,
            edited,
            path,
            base_model=base_model,
            revision=revision,
            machaon_metadata=metadata,
            form=form,
            rank=rank,
        )

    # -- WSP evidence ---------------------------------------------------

    def _write_state(self, deltas: Dict[str, np.ndarray]) -> None:
        """Temporarily write a delta map on top of the saved originals."""
        for name, original in self._originals.items():
            delta = deltas.get(name)
            self.runtime.write(name, original if delta is None else original + delta)

    def evaluate_wsp(
        self,
        prompts: Sequence[str],
        targets: Sequence[Any],
        seed: int = 0,
        controls: Optional[Dict[str, np.ndarray]] = None,
        include_base: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate candidate and a matched random control on the same model state.

        The candidate state is restored before returning, even if a control
        evaluation raises.  This method never changes the edit budget or the
        saved originals; it is a read-only evidence pass over the current edit.
        """
        if not self._originals:
            raise RuntimeError("no applied edit; evaluate_wsp needs a candidate state")

        candidate_deltas = self.changes()
        control_deltas = controls or random_controls(candidate_deltas, seed=seed)
        result: Dict[str, Dict[str, Any]] = {
            "candidate": self.evaluate(prompts, targets),
            "controls": {},
        }
        try:
            if include_base:
                self._write_state({})
                result["base"] = self.evaluate(prompts, targets)
            self._write_state(control_deltas)
            result["controls"]["random_same_support"] = self.evaluate(prompts, targets)
        finally:
            self._write_state(candidate_deltas)
        return result

    def run_headroom(
        self,
        evaluate_fn: Callable[[Any], Mapping[str, Any]],
        name: str = "headroom",
    ) -> Dict[str, Any]:
        """Evaluate the *unaltered base* with a caller-supplied evaluator.

        This is the bridge for a headroom baseline that is not expressible as
        next-token logits: a different decoding mode, a larger token budget, or
        a benchmark-specific scorer.  The candidate edit is restored in a
        ``finally`` block, so a failed baseline cannot leave the session in the
        wrong state.
        """
        if not self._originals:
            raise RuntimeError("no applied edit; run_headroom needs a candidate state")
        candidate_deltas = self.changes()
        try:
            self._write_state({})
            metrics = dict(evaluate_fn(self.runtime))
        finally:
            self._write_state(candidate_deltas)
        metrics.setdefault("name", name)
        return metrics

    def assess_wsp(
        self,
        prompts: Sequence[str],
        targets: Sequence[Any],
        headroom: Optional[Dict[str, Any]] = None,
        headroom_runner: Optional[Callable[[Any], Mapping[str, Any]]] = None,
        seed: int = 0,
        controls: Optional[Dict[str, np.ndarray]] = None,
        protocol=None,
        target_metric: str = "accuracy",
        mechanism: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run the WSP decision layer over the current candidate edit.

        The session-level pass has target/KL/perplexity evidence, not a full
        benchmark panel.  It therefore defaults to ``require_gates=False`` and
        reports the missing gate metrics explicitly; use the CLI/MCP ``wsp``
        audit with a complete metrics report when gate acceptance is required.
        Pass ``mechanism`` from :func:`machaon.generation_mechanism` or another
        mechanism evaluator to make obstruction relief part of acceptance.
        """
        evidence = self.evaluate_wsp(
            prompts, targets, seed=seed, controls=controls, include_base=True
        )
        if headroom is None and headroom_runner is not None:
            headroom = self.run_headroom(headroom_runner)
        protocol = protocol or WSPProtocol(
            target_metric=target_metric, require_gates=False
        )
        return assess_wsp(
            evidence["base"],
            evidence["candidate"],
            controls=evidence["controls"],
            headroom=headroom,
            protocol=protocol,
            mechanism=mechanism,
        )

    def assess_generation_wsp(
        self,
        prompts: Sequence[str],
        expected: Sequence[Any],
        *,
        headroom_config: Optional[DecodingConfig] = None,
        config: Optional[DecodingConfig] = None,
        match: str = "exact",
        seed: int = 0,
        controls: Optional[Dict[str, np.ndarray]] = None,
        protocol=None,
        mechanism_protocol: Optional[MechanismProtocol] = None,
    ) -> Dict[str, Any]:
        """Evaluate a generation task with target and mechanism evidence.

        Candidate, unedited base, matched random control and optional headroom
        are generated under the same decoding contract.  Their stop traces feed
        the mechanism gate, while generated text feeds accuracy.  The candidate
        state is restored in a ``finally`` block even if generation fails.
        """
        if not self._originals:
            raise RuntimeError(
                "no applied edit; assess_generation_wsp needs a candidate state"
            )
        if not hasattr(self.runtime, "generate_records"):
            raise RuntimeError("runtime does not expose generate_records")
        if len(prompts) != len(expected):
            raise ValueError("prompts and expected answers differ in length")
        prompts = list(prompts)
        expected = list(expected)
        if not prompts:
            raise ValueError("assess_generation_wsp needs at least one prompt")
        config = config or DecodingConfig()
        candidate_deltas = self.changes()
        control_deltas = controls or random_controls(candidate_deltas, seed=seed)
        records: Dict[str, Any] = {}

        def run_generation(decoding: DecodingConfig) -> list:
            return self.runtime.generate_records(prompts, **decoding.kwargs())

        def target_report(rows: Sequence[Mapping[str, Any]],
                          decoding: DecodingConfig) -> Dict[str, Any]:
            texts = [str(row.get("text", "")) for row in rows]
            scored = score_generation_texts(texts, expected, match)
            return {
                "accuracy": scored["accuracy"],
                "n": scored["n"],
                "decoding": decoding.to_dict(),
                "match": match,
            }

        try:
            records["candidate"] = run_generation(config)
            self._write_state({})
            records["base"] = run_generation(config)
            if headroom_config is not None:
                records["headroom"] = run_generation(headroom_config)
            self._write_state(control_deltas)
            records["control"] = run_generation(config)
        finally:
            self._write_state(candidate_deltas)

        mechanism_protocol = mechanism_protocol or MechanismProtocol()
        mechanism = generation_mechanism(
            records["base"],
            records["candidate"],
            controls={"random_same_support": records["control"]},
            headroom=records.get("headroom"),
        )
        mechanism["protocol"] = mechanism_protocol.to_dict()
        protocol = protocol or WSPProtocol(
            target_metric="accuracy", require_gates=False
        )
        base = target_report(records["base"], config)
        candidate = target_report(records["candidate"], config)
        headroom = (
            target_report(records["headroom"], headroom_config)
            if headroom_config is not None else None
        )
        control_report = target_report(records["control"], config)
        return assess_wsp(
            base,
            candidate,
            controls={"random_same_support": control_report},
            headroom=headroom,
            protocol=protocol,
            mechanism=mechanism,
        )

    def search_layers(
        self,
        layers: Sequence[int],
        proposal_factory: Callable[["Session", int], Any],
        evaluation: Callable[["Session"], Mapping[str, Any]],
        *,
        metric: str = "accuracy",
        higher_is_better: bool = True,
        max_evaluations: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Apply, evaluate and roll back a proposal at each candidate layer.

        The session must be clean before the search starts. Every trial is
        cleaned up in a ``finally`` path; a cleanup failure aborts the search
        rather than continuing from a dirty state. The returned evidence keeps
        per-layer metrics and failures, not just the winner.
        """
        if self._originals:
            raise RuntimeError(
                "search_layers needs a clean session; roll back existing edits first"
            )
        if self.state().has_pending:
            raise RuntimeError(
                "search_layers needs no pending proposal; apply or discard it first"
            )

        def evaluate_layer(layer: int) -> Mapping[str, Any]:
            proposal_factory(self, int(layer))
            if not self.state().has_pending:
                raise RuntimeError("proposal_factory did not create a pending proposal")
            applied = self.apply()
            metrics = dict(evaluation(self))
            metrics["relative_delta"] = applied["relative_delta"]
            metrics["budget_total"] = applied["budget_total"]
            return metrics

        return _search_layers(
            layers,
            evaluate_layer,
            metric=metric,
            higher_is_better=higher_is_better,
            max_evaluations=max_evaluations,
            cleanup=self.rollback,
        )

    def search_rome_layers(
        self,
        prompt: str,
        target: str,
        layers: Sequence[int],
        *,
        solver: str = "direction",
        steps: int = 250,
        lr: float = 0.02,
        max_rel_norm: Optional[float] = None,
        preserve_prompts: Sequence[str] = (),
        metric: str = "accuracy",
        evaluate_prompts: Optional[Sequence[str]] = None,
        evaluate_targets: Optional[Sequence[Any]] = None,
        max_evaluations: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Live-search ROME edit sites, evaluating and rolling back each layer."""
        prompts = list(evaluate_prompts) if evaluate_prompts is not None else [prompt]
        targets = list(evaluate_targets) if evaluate_targets is not None else [target]

        def propose(session: "Session", layer: int) -> None:
            session.propose(
                prompt,
                target,
                int(layer),
                solver=solver,
                steps=int(steps),
                lr=float(lr),
                max_rel_norm=max_rel_norm,
                preserve_prompts=preserve_prompts,
            )

        report = self.search_layers(
            layers,
            propose,
            lambda session: session.evaluate(prompts, targets),
            metric=metric,
            max_evaluations=max_evaluations,
        )
        report["search"] = "rome"
        report["prompt"] = prompt
        report["target"] = target
        report["solver"] = solver
        return report

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
