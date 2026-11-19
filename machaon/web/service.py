"""Transport-free operations used by the local web API.

The service is intentionally a thin composition layer.  It calls the same
``machaon.tools`` functions and the same ``machaon.session.Session`` methods as
the CLI and MCP server.  FastAPI handlers only parse HTTP and attach a job.
"""
from __future__ import annotations

import importlib
from dataclasses import asdict, is_dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional, Sequence

from ..budget import EditBudget
from .security import PathPolicy

__all__ = ["WebService"]


def _session_state(session: Any) -> Dict[str, Any]:
    state = session.state()
    if is_dataclass(state):
        return asdict(state)
    if isinstance(state, dict):
        return dict(state)
    if hasattr(state, "__dict__"):
        return dict(vars(state))
    return {"state": str(state)}


def _default_session_factory(**kwargs):
    from ..session import Session

    return Session.open(**kwargs)


class WebService:
    """One loaded session plus stateless checkpoint operations."""

    def __init__(
        self,
        policy: PathPolicy,
        tools_module: Optional[Any] = None,
        session_factory: Optional[Any] = None,
        intent_module: Optional[Any] = None,
        recipe: Optional[Any] = None,
        fetch_module: Optional[Any] = None,
        many_recipe: Optional[Any] = None,
        program_runner: Optional[Any] = None,
    ) -> None:
        self.policy = policy
        self.tools = tools_module or importlib.import_module("machaon.tools")
        self.session_factory = session_factory or _default_session_factory
        self.intent = intent_module or importlib.import_module("machaon.intent")
        self.recipe = recipe
        self.many_recipe = many_recipe
        self.program_runner = program_runner
        self.fetch = fetch_module or importlib.import_module("machaon.fetch")
        self.session: Optional[Any] = None
        self._open_options: Optional[Dict[str, Any]] = None
        self._lock = RLock()

    # -- checkpoint inspection ------------------------------------------

    def describe_checkpoint(self, path: str) -> Dict[str, Any]:
        return self.tools.describe_checkpoint(self.policy.require_existing(path))

    def list_tensors(
        self,
        path: str,
        contains: Optional[str] = None,
        suffix: Optional[str] = None,
        layer: Optional[int] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        return self.tools.list_tensors(
            self.policy.require_existing(path),
            contains=contains,
            suffix=suffix,
            layer=layer,
            limit=limit,
        )

    def audit_checkpoint_edit(
        self,
        base_path: str,
        edit_dir: str,
        max_rel_norm: Optional[float] = None,
    ) -> Dict[str, Any]:
        return self.tools.audit_checkpoint_edit(
            self.policy.require_existing(base_path),
            self.policy.require_existing(edit_dir),
            max_rel_norm=max_rel_norm,
        )

    def apply_delta(
        self,
        base_path: str,
        tensor: str,
        delta_path: str,
        out_dir: str,
        refuse_rescale: bool = True,
    ) -> Dict[str, Any]:
        return self.tools.apply_delta(
            self.policy.require_existing(base_path),
            tensor,
            self.policy.require_existing(delta_path),
            self.policy.require(out_dir),
            refuse_rescale=refuse_rescale,
        )

    def gate_metrics(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        return self.tools.gate_metrics(metrics)

    def wsp_audit(self, report: Dict[str, Any]) -> Dict[str, Any]:
        return self.tools.wsp_audit(report)

    def save_wsp_evidence(
        self,
        out_dir: str,
        report: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        path = self.policy.require(out_dir)
        return self.tools.save_wsp_evidence(path, report, metadata)

    def assess_session_wsp(
        self,
        prompts: Sequence[str],
        targets: Sequence[Any],
        headroom: Optional[Dict[str, Any]] = None,
        seed: int = 0,
        target_metric: str = "accuracy",
        mechanism: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        session = self._require_session()
        kwargs: Dict[str, Any] = {
            "headroom": headroom,
            "seed": seed,
            "target_metric": target_metric,
        }
        if mechanism is not None:
            kwargs["mechanism"] = mechanism
        return session.assess_wsp(prompts, targets, **kwargs)

    def assess_session_evidence(
        self,
        out_dir: str,
        prompts: Sequence[str],
        targets: Sequence[Any],
        headroom: Optional[Dict[str, Any]] = None,
        seed: int = 0,
        target_metric: str = "accuracy",
        mechanism: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        session = self._require_session()
        kwargs: Dict[str, Any] = {
            "headroom": headroom,
            "seed": seed,
            "target_metric": target_metric,
        }
        if mechanism is not None:
            kwargs["mechanism"] = mechanism
        report = session.assess_wsp(prompts, targets, **kwargs)
        path = self.policy.require(out_dir)
        evidence = self.tools.save_wsp_evidence(
            path,
            report,
            metadata={
                "model": getattr(session, "model_id", None),
                "target_metric": target_metric,
                "seed": int(seed),
            },
        )
        return {
            "report": report,
            "evidence": evidence,
            "state": self.state(),
        }

    def export_session_edit(
        self,
        out_dir: str,
        evidence_sha256: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        session = self._require_session()
        path = self.policy.require(out_dir)
        link = dict(metadata or {})
        if evidence_sha256:
            link["evidence_sha256"] = evidence_sha256
            link["evidence_format"] = "machaon.wsp.evidence"
        manifest = session.export_openwsp(
            path,
            base_model=getattr(session, "model_id", None),
            metadata=link or None,
        )
        return {
            "output": path,
            "manifest": manifest,
            "state": self.state(),
        }

    def verify_binding(self, edit_dir: str, evidence_dir: str) -> Dict[str, Any]:
        return self.tools.verify_wsp_binding(
            self.policy.require_existing(edit_dir),
            self.policy.require_existing(evidence_dir),
        )

    # -- model and session ----------------------------------------------

    def open_session(
        self,
        path: str,
        dtype: str = "auto",
        device: Optional[str] = None,
        quantize: Optional[str] = None,
        budget_total: Optional[float] = 0.10,
    ) -> Dict[str, Any]:
        resolved = self.policy.require_existing(path)
        with self._lock:
            if self.session is not None:
                state = _session_state(self.session)
                if state.get("edited"):
                    raise RuntimeError(
                        "the current session has applied edits; roll them back "
                        "before opening another checkpoint"
                    )
            budget = None
            if budget_total is not None:
                budget = EditBudget(max_total=float(budget_total))
            self.session = self.session_factory(
                model_path=resolved,
                dtype=dtype,
                device=device,
                quantize=quantize,
                budget=budget,
            )
            self._open_options = {
                "dtype": dtype,
                "device": device,
                "quantize": quantize,
            }
            return self.state()

    def close_session(self, force: bool = False) -> Dict[str, Any]:
        with self._lock:
            if self.session is None:
                return {"ready": False}
            state = _session_state(self.session)
            if state.get("edited") and not force:
                raise RuntimeError(
                    "the session has applied edits; roll them back or set force=true"
                )
            if force and state.get("edited"):
                self.session.rollback()
            self.session = None
            self._open_options = None
            return {"ready": False}

    def state(self) -> Dict[str, Any]:
        with self._lock:
            if self.session is None:
                return {"ready": False}
            payload = _session_state(self.session)
            payload["ready"] = True
            return payload

    def baseline(
        self,
        neutral_prompts: Sequence[str],
        corpus_texts: Sequence[str] = (),
    ) -> Dict[str, Any]:
        session = self._require_session()
        return {
            "baseline": session.baseline(neutral_prompts, corpus_texts),
            "state": self.state(),
        }

    def propose(self, **kwargs: Any) -> Dict[str, Any]:
        session = self._require_session()
        densify_layer = kwargs.pop("densify_from_layer", False)
        densified = None
        if densify_layer:
            runtime = getattr(session, "runtime", None)
            if runtime is None or not hasattr(runtime, "densify_from"):
                raise RuntimeError(
                    "this session runtime does not expose densify_from"
                )
            densified = runtime.densify_from(int(kwargs["layer"]))
        payload = {
            "proposal": session.propose(**kwargs),
            "state": self.state(),
        }
        if densified is not None:
            payload["densified"] = densified
        return payload

    def propose_batch(self, **kwargs: Any) -> Dict[str, Any]:
        session = self._require_session()
        return {
            "proposal": session.propose_batch(**kwargs),
            "state": self.state(),
        }

    def apply(self) -> Dict[str, Any]:
        session = self._require_session()
        return {
            "applied": session.apply(),
            "state": self.state(),
        }

    def evaluate(
        self,
        prompts: Sequence[str],
        targets: Sequence[Any],
        gate: bool = True,
    ) -> Dict[str, Any]:
        session = self._require_session()
        metrics = session.evaluate(prompts, targets)
        payload: Dict[str, Any] = {
            "metrics": metrics,
            "state": self.state(),
        }
        if gate:
            payload["gate"] = self.tools.gate_metrics(metrics)
        return payload

    # -- one-sentence teaching -----------------------------------------

    def preview_teach(self, sentence: str) -> Dict[str, Any]:
        """Compile a sentence without touching a model."""
        intent = self.intent.parse_intent(sentence)
        payload = intent.to_dict()
        payload["command"] = intent.to_command()
        return payload

    def teach(
        self,
        sentence: Optional[str] = None,
        prompt: Optional[str] = None,
        answer: Optional[str] = None,
        layers: Optional[Sequence[int]] = None,
        solver: str = "direction",
        max_rel_norm: float = 0.02,
        dry_run: bool = False,
        out_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the teaching recipe on the open session.

        When ``out_dir`` is supplied the accepted checkpoint is written before
        returning.  A write failure rolls the in-memory edit back, so a failed
        save never leaves a half-persisted run.
        """
        session = self._require_session()
        if sentence:
            intent = self.intent.parse_intent(sentence)
            prompt, answer = intent.prompt, intent.answer
            intent_payload = intent.to_dict()
        else:
            if not prompt or answer is None or answer == "":
                raise ValueError("teach needs a sentence, or both prompt and answer")
            intent_payload = {
                "kind": "teach",
                "prompt": prompt,
                "answer": answer,
                "source": "arguments",
                "confidence": "high",
                "notes": [],
            }
        from ..recipes import TeachPolicy

        recipe = self.recipe
        if recipe is None:
            from ..recipes import teach_fact

            recipe = teach_fact
        result = recipe(
            session,
            prompt,
            answer,
            layers=list(layers) if layers is not None else None,
            policy=TeachPolicy(max_rel_norm=float(max_rel_norm)),
            solver=solver,
            dry_run=bool(dry_run),
        )
        payload = {
            "intent": intent_payload,
            "result": result.to_dict(),
            "state": self.state(),
        }
        if result.status == "applied" and out_dir:
            try:
                payload["saved"] = self.save_checkpoint(out_dir)
            except Exception as exc:
                try:
                    session.rollback()
                except Exception as rollback_exc:
                    raise RuntimeError(
                        "saving the checkpoint failed (%s) and rollback also "
                        "failed (%s)" % (exc, rollback_exc)
                    ) from exc
                raise
            payload["state"] = self.state()
        return payload

    def rollback(self) -> Dict[str, Any]:
        session = self._require_session()
        return {
            "rolled_back": session.rollback(),
            "state": self.state(),
        }

    def save_checkpoint(self, out_dir: str) -> Dict[str, Any]:
        """Persist applied session deltas through the checkpoint adapter."""
        session = self._require_session()
        state = _session_state(session)
        edited = list(state.get("edited", []))
        if not edited:
            raise RuntimeError("the session has no applied edits to save")
        out = Path(self.policy.require(out_dir))
        if out.exists():
            raise RuntimeError(
                "output already exists: %s (choose another path or remove it)" % out
            )
        from ..checkpointing import save_session

        saved = save_session(session, out, require_new=True)
        return {
            "output": saved["output"],
            "edited": edited,
            "checkpoint": saved["checkpoint"],
            "state": self.state(),
        }

    def teach_many(
        self,
        facts: Sequence[str],
        out_dir: str,
        budget: float = 0.10,
        max_rel_norm: float = 0.02,
        keep_steps: bool = False,
        continue_on_reject: bool = False,
    ) -> Dict[str, Any]:
        """Run the multi-fact chain on the open checkpoint.

        The current session must be clean.  The original session is left on the
        base checkpoint; the chain writes a separate output container.
        """
        session = self._require_session()
        state = _session_state(session)
        if state.get("edited") or state.get("has_pending"):
            raise RuntimeError(
                "teach-many needs a clean session; roll back applied edits "
                "and clear the pending proposal first"
            )
        facts = [str(item) for item in facts]
        if not facts:
            raise ValueError("teach-many needs at least one fact")
        out = Path(self.policy.require(out_dir))
        if out.exists():
            raise RuntimeError("output already exists: %s" % out)
        options = dict(self._open_options or {
            "dtype": "auto", "device": None, "quantize": None,
        })

        def factory(path, step_budget):
            return self.session_factory(
                model_path=str(path),
                budget=step_budget,
                **options,
            )

        from ..recipes import TeachPolicy

        recipe = self.many_recipe
        if recipe is None:
            from ..teach_many import teach_many as recipe
        result = recipe(
            factory,
            session.model_id,
            facts,
            out,
            budget=float(budget),
            policy=TeachPolicy(max_rel_norm=float(max_rel_norm)),
            keep_steps=bool(keep_steps),
            continue_on_reject=bool(continue_on_reject),
        )
        payload = result.to_dict() if hasattr(result, "to_dict") else dict(result)
        return {"result": payload, "state": self.state()}

    def check_saved(
        self,
        path: str,
        sentence: Optional[str] = None,
        prompt: Optional[str] = None,
        answer: Optional[str] = None,
        facts: Optional[Sequence[str]] = None,
        dtype: str = "auto",
        device: Optional[str] = None,
        quantize: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate one fact against a saved checkpoint without editing it."""
        resolved = self.policy.require_existing(path)
        if quantize is not None and dtype != "auto":
            raise ValueError("quantize requires dtype='auto'")
        if facts is not None:
            if sentence or prompt is not None or answer is not None:
                raise ValueError("use facts or sentence/prompt+answer, not both")
            if not facts:
                raise ValueError("check needs at least one fact")
            pairs = []
            for index, item in enumerate(facts, 1):
                try:
                    intent = self.intent.parse_intent(str(item))
                except Exception as exc:
                    raise ValueError("fact %d: %s" % (index, exc)) from exc
                pairs.append((intent.prompt, intent.answer))
        elif sentence:
            intent = self.intent.parse_intent(sentence)
            prompt, answer = intent.prompt, intent.answer
            pairs = [(prompt, answer)]
        elif not prompt or answer is None or answer == "":
            raise ValueError("check needs a sentence, or both prompt and answer")
        else:
            pairs = [(prompt, answer)]

        from ..check import evaluate_fact

        current = self.state() if self.session is not None else {"ready": False}
        owned = False
        session = None
        try:
            if self.session is not None:
                same_model = False
                model_path = current.get("model")
                if model_path:
                    if Path(model_path).resolve() == Path(resolved).resolve():
                        same_model = True
                if not same_model:
                    raise RuntimeError(
                        "close the current session before checking a different checkpoint"
                    )
                if current.get("edited"):
                    raise RuntimeError(
                        "roll back applied edits before checking the saved checkpoint"
                    )
                session = self.session
            else:
                session = self.session_factory(
                    model_path=resolved,
                    dtype=dtype,
                    device=device,
                    quantize=quantize,
                    budget=None,
                )
                owned = True

            if facts is not None:
                results = []
                for index, (item_prompt, item_answer) in enumerate(pairs, 1):
                    item = evaluate_fact(session, item_prompt, item_answer)
                    item["index"] = index
                    item["model"] = resolved
                    results.append(item)
                return {
                    "passed": all(item["passed"] for item in results),
                    "model": resolved,
                    "facts": results,
                }
            result = evaluate_fact(session, pairs[0][0], pairs[0][1])
        finally:
            if owned:
                session = None

        result["model"] = resolved
        return result

    # -- portable programs ---------------------------------------------

    def _program_from_text(self, text: str, base_dir: Optional[str] = None):
        from pathlib import Path

        from ..program import parse_program

        if not isinstance(text, str) or not text.strip():
            raise ValueError("program text is required")
        if base_dir:
            base = Path(self.policy.require_existing(base_dir))
        else:
            base = self.policy.roots[0]
        if not base.is_dir():
            raise ValueError("program base must be a directory: %s" % base)
        program = parse_program(
            text, source="<web program>", base_dir=base
        )
        # A pasted program is still subject to the same filesystem allowlist
        # as every other browser operation.
        self.policy.require(program.model)
        self.policy.require(program.output)
        return program, base

    def plan_program(
        self, text: str, base_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Parse and resolve a program without downloading or loading a model."""
        from ..program import program_preflight

        program, base = self._program_from_text(text, base_dir)
        payload = program.to_dict()
        preflight = program_preflight(program)
        payload["base_dir"] = str(base)
        payload["preflight"] = preflight
        payload["model_exists"] = preflight["model_exists"]
        return payload

    def new_program(
        self,
        facts: Sequence[str],
        *,
        model: str,
        output: Optional[str] = None,
        fetch: Optional[str] = None,
        name: str = "",
        protect: Optional[Sequence[str]] = None,
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
    ) -> Dict[str, Any]:
        """Generate the shared text format without writing a file."""
        from ..program import parse_program, render_program_text

        output_text = output or model.rstrip("/\\") + "-machaon"
        text = render_program_text(
            model=model,
            output=output_text,
            fetch=fetch,
            name=name,
            teach=list(facts),
            protect=list(protect or ()),
            budget=budget,
            max_rel_norm=max_rel_norm,
            max_kl=max_kl,
            min_panel_agreement=min_panel_agreement,
            layers=layers,
            solver=solver,
            steps=steps,
            lr=lr,
            max_layers=max_layers,
            dtype=dtype,
            device=device,
            quantize=quantize,
            verify=verify,
            continue_on_reject=continue_on_reject,
            keep_steps=keep_steps,
        )
        program = parse_program(
            text,
            source="<web generated program>",
            base_dir=self.policy.roots[0],
        )
        return {
            "text": text,
            "program": program.to_dict(),
        }

    def run_program(
        self,
        text: str,
        base_dir: Optional[str] = None,
        progress: Optional[Any] = None,
        report_out: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a pasted program through the shared program runner."""
        program, base = self._program_from_text(text, base_dir)
        report_path = None
        if report_out:
            candidate = Path(report_out).expanduser()
            if not candidate.is_absolute():
                candidate = base / candidate
            report_path = self.policy.require(str(candidate))
        runner = self.program_runner
        if runner is None:
            from ..program import run_program as runner
        kwargs: Dict[str, Any] = {}
        if progress is not None:
            kwargs["progress"] = progress
        if report_path is not None:
            kwargs["report_out"] = report_path
        result = runner(program, **kwargs)
        if hasattr(result, "to_dict"):
            return dict(result.to_dict())
        return dict(result)

    def verify_program_report(
        self,
        report_text: str,
        program_text: Optional[str] = None,
        check_model: bool = False,
        checkpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Verify a pasted report without trusting paths outside the allowlist."""
        from ..program import (
            parse_program_report,
            program_report_checkpoint,
            verify_program_report,
        )

        if not isinstance(report_text, str) or not report_text.strip():
            raise ValueError("report text is required")
        document = parse_program_report(
            report_text, source="<web program report>"
        )
        if program_text is not None and str(program_text).strip():
            from ..program import parse_program

            actual = parse_program(
                str(program_text),
                source="<web program>",
                base_dir=self.policy.roots[0],
            ).sha256
            expected = document["program"]["sha256"]
            if actual != expected:
                raise ValueError(
                    "program hash does not match the report: expected %s, got %s"
                    % (expected, actual)
                )
        if checkpoint and not check_model:
            raise ValueError("checkpoint can only be used with check_model")
        target = None
        if checkpoint:
            target = self.policy.require_existing(checkpoint)
        elif check_model:
            target = self.policy.require_existing(
                program_report_checkpoint(document)
            )
        result = verify_program_report(
            document, check_model=check_model, checkpoint=target
        )
        if program_text is not None and str(program_text).strip():
            result["program_hash_matches"] = True
        if target is not None:
            # The allowlist check and the verifier see the same saved path.
            result["model_check"]["checkpoint"] = target
        return result

    def _require_session(self) -> Any:
        with self._lock:
            if self.session is None:
                raise RuntimeError("no model is open; open a checkpoint first")
            return self.session

    # -- discovery ------------------------------------------------------

    def models(self, limit: int = 200) -> Dict[str, Any]:
        """List checkpoint-looking entries in the allowed roots.

        This is a convenience picker, not an indexer.  It reads directory
        entries and marker names only; it never loads a weight tensor.
        """
        markers = ("config.json", "model.safetensors.index.json",
                   "model.safetensors", "pytorch_model.bin")
        found: List[Dict[str, Any]] = []
        seen = set()

        def add(path: Path) -> None:
            key = str(path)
            if key in seen or len(found) >= max(1, int(limit)):
                return
            seen.add(key)
            found.append({
                "path": key,
                "name": path.name or str(path),
                "kind": "directory" if path.is_dir() else "file",
            })

        for root in self.policy.roots:
            try:
                if root.is_file() and root.suffix == ".safetensors":
                    add(root)
                    continue
                if root.is_dir() and self._looks_like_checkpoint(root, markers):
                    add(root)
                if not root.is_dir():
                    continue
                for child in sorted(root.iterdir(), key=lambda item: item.name.lower()):
                    if child.name.startswith("."):
                        continue
                    if len(found) >= limit:
                        break
                    if child.is_dir() and self._looks_like_checkpoint(child, markers):
                        add(child)
                    elif child.is_file() and child.suffix == ".safetensors":
                        add(child)
            except OSError:
                continue
        return {"roots": [str(root) for root in self.policy.roots], "models": found}

    @staticmethod
    def capabilities() -> Dict[str, bool]:
        """Report whether the optional real-model runtimes are importable."""
        import importlib.util

        found = {}
        for name in ("torch", "transformers", "safetensors"):
            try:
                found[name] = importlib.util.find_spec(name) is not None
            except (ImportError, ValueError):
                found[name] = False
        return found

    @staticmethod
    def devices() -> Dict[str, Any]:
        """Which compute devices this machine can actually use."""
        from ..devices import device_report

        return device_report()

    def fetch_model(
        self,
        starter: str,
        dest: str,
        endpoint: Optional[str] = None,
        progress: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Download one recommended starter checkpoint below an allowed root.

        The browser can request only the fixed starter catalogue, never an
        arbitrary repository, so this endpoint cannot be repointed at an
        unrelated host or model.
        """
        recommended = getattr(self.fetch, "RECOMMENDED", {})
        item = recommended.get(starter)
        if item is None:
            raise ValueError(
                "unknown starter %r; choose one of %s"
                % (starter, ", ".join(sorted(recommended)))
            )
        path = Path(self.policy.require(dest))
        if path.exists() and not path.is_dir():
            raise RuntimeError(
                "download destination is a file, not a directory: %s" % path
            )
        # Existing files are reused so an interrupted large download can be
        # resumed by pressing the same button again.
        report = self.fetch.fetch_model(
            item["repo"], path, endpoint=endpoint, require_new=False,
            progress=progress,
        )
        report["starter"] = starter
        report["name"] = item.get("name", starter)
        return report

    @staticmethod
    def _looks_like_checkpoint(path: Path, markers: Sequence[str]) -> bool:
        for marker in markers:
            if (path / marker).exists():
                return True
        try:
            return any(path.glob("*.safetensors"))
        except OSError:
            return False
