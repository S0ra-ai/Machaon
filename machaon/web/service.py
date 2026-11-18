"""Transport-free operations used by the local web API.

The service is intentionally a thin composition layer.  It calls the same
``machaon.tools`` functions and the same ``machaon.session.Session`` methods as
the CLI and MCP server.  FastAPI handlers only parse HTTP and attach a job.
"""
from __future__ import annotations

import importlib
import shutil
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
    ) -> None:
        self.policy = policy
        self.tools = tools_module or importlib.import_module("machaon.tools")
        self.session_factory = session_factory or _default_session_factory
        self.intent = intent_module or importlib.import_module("machaon.intent")
        self.recipe = recipe
        self.fetch = fetch_module or importlib.import_module("machaon.fetch")
        self.session: Optional[Any] = None
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
        source = Path(session.model_id)
        out = Path(self.policy.require(out_dir))
        if out.exists():
            raise RuntimeError(
                "output already exists: %s (choose another path or remove it)" % out
            )
        from ..adapters import Checkpoint

        checkpoint = Checkpoint(source)
        changes = session.changes()
        stored = getattr(session.runtime, "stored_values", None)
        try:
            # A single 27B edit is one shard; avoid rewriting every shard when
            # the session contains exactly one applied tensor.
            if len(changes) == 1:
                name = next(iter(changes))
                values = session.runtime.read(name)
                if callable(stored):
                    values = stored(name, values)
                report = checkpoint.write_tensor(
                    name, values, out
                )
            else:
                if callable(stored):
                    changes = {
                        name: stored(name, delta) for name, delta in changes.items()
                    }
                report = checkpoint.write_with(changes, out)
        except Exception:
            shutil.rmtree(out, ignore_errors=True)
            raise
        return {
            "output": str(out),
            "edited": edited,
            "checkpoint": report,
            "state": self.state(),
        }

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

    def fetch_model(
        self,
        starter: str,
        dest: str,
        endpoint: Optional[str] = None,
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
            item["repo"], path, endpoint=endpoint, require_new=False
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
