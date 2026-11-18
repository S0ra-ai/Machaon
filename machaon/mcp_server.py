"""MCP stdio server exposing file-level and model-level operations.

    python -m machaon.mcp_server

The transport is deliberately thin.  File-level tools call plain functions in
:mod:`machaon.tools` and never load a model.  Model-level tools operate on one
explicitly opened :class:`machaon.session.Session`; nothing is written until
``apply_edit`` is called.

Requires the optional extra: ``pip install "machaon[mcp]"``.
"""
from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

__all__ = ["create_server", "main"]


def _server_class():
    """The MCP server class across SDK generations.

    ``mcp`` 1.x exposes ``FastMCP``; 2.x renamed it to ``MCPServer`` but kept
    the ``@tool()`` decorator and ``run(transport="stdio")``. Supporting both
    avoids pinning users to an old SDK.
    """
    try:
        from mcp.server.fastmcp import FastMCP

        return FastMCP
    except ImportError:
        pass
    try:
        from mcp.server.mcpserver import MCPServer

        return MCPServer
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            'the MCP server needs the optional extra: pip install "machaon[mcp]"'
        ) from exc


def create_server():
    from . import tools

    server = _server_class()("Machaon")
    session_box: Dict[str, Any] = {"session": None}

    NO_MODEL = {"error": "no model is open; call open_model first"}

    def current():
        return session_box["session"]

    @server.tool()
    def describe_checkpoint(path: str) -> Dict[str, Any]:
        """Tensor count, parameter count, dtype mix and shards. Header only."""
        return tools.describe_checkpoint(path)

    @server.tool()
    def list_tensors(path: str, contains: Optional[str] = None,
                     suffix: Optional[str] = None, layer: Optional[int] = None,
                     limit: int = 50) -> List[Dict[str, Any]]:
        """List tensor names, shapes and dtypes. Never returns values."""
        return tools.list_tensors(path, contains, suffix, layer, limit)

    @server.tool()
    def audit_edit(base_path: str, edit_dir: str,
                   max_rel_norm: Optional[float] = None) -> Dict[str, Any]:
        """Verify an OpenWSP v1 edit against a base checkpoint and report drift."""
        return tools.audit_checkpoint_edit(base_path, edit_dir, max_rel_norm)

    @server.tool()
    def apply_delta(base_path: str, tensor: str, delta_path: str,
                    out_dir: str) -> Dict[str, Any]:
        """Apply one .npy delta to one tensor, writing a new checkpoint."""
        return tools.apply_delta(base_path, tensor, delta_path, out_dir)

    @server.tool()
    def gate(metrics: Dict[str, float]) -> Dict[str, Any]:
        """Apply the acceptance gates. Missing metrics are failures."""
        return tools.gate_metrics(metrics)

    @server.tool()
    def wsp_audit(report: Dict[str, Any]) -> Dict[str, Any]:
        """Classify a WSP report as repair, acquisition or non-specific."""
        return tools.wsp_audit(report)

    @server.tool()
    def wsp_save(report: Dict[str, Any], out_dir: str,
                 metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Write a content-addressed WSP evidence bundle."""
        return tools.save_wsp_evidence(out_dir, report, metadata)

    @server.tool()
    def verify_binding(edit_dir: str, evidence_dir: str) -> Dict[str, Any]:
        """Verify an OpenWSP edit and WSP evidence bundle are bound together."""
        return tools.verify_wsp_binding(edit_dir, evidence_dir)

    @server.tool()
    def plan_surgery(failure_type: str,
                     layer_scores: Optional[Dict[str, float]] = None,
                     patch_report: Optional[Dict[str, Any]] = None,
                     module_suffix: str = "mlp.down_proj",
                     max_rel_norm: float = 0.02,
                     min_score: float = 0.0) -> Dict[str, Any]:
        """Choose edit, diagnose or do-not-edit from failure and site evidence."""
        return tools.plan_surgery(
            failure_type,
            layer_scores=layer_scores,
            patch_report=patch_report,
            module_suffix=module_suffix,
            max_rel_norm=max_rel_norm,
            min_score=min_score,
        )

    @server.tool()
    def diagnose_failure(base: Dict[str, Any],
                         headroom: Optional[Dict[str, Any]] = None,
                         mechanism: Optional[Dict[str, Any]] = None,
                         target_metric: str = "accuracy",
                         min_target_gap: float = 0.05,
                         min_mechanism_gap: float = 0.05) -> Dict[str, Any]:
        """Classify obstruction vs capability gap without editing weights."""
        return tools.diagnose_failure(
            base,
            headroom=headroom,
            mechanism=mechanism,
            target_metric=target_metric,
            min_target_gap=min_target_gap,
            min_mechanism_gap=min_mechanism_gap,
        )

    @server.tool()
    def parse_intent(sentence: str) -> Dict[str, Any]:
        """Compile a plain-language teaching request into an exact contract."""
        return tools.parse_intent(sentence)

    @server.tool()
    def check_saved(model: str, sentence: Optional[str] = None,
                    prompt: Optional[str] = None, answer: Optional[str] = None,
                    facts: Optional[List[str]] = None,
                    dtype: str = "auto", device: Optional[str] = None,
                    quantize: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate one fact against a saved checkpoint without opening a session."""
        from .check import check_checkpoint, check_facts

        try:
            if facts is not None:
                return check_facts(
                    model,
                    facts,
                    dtype=dtype,
                    device=device,
                    quantize=quantize,
                )
            return check_checkpoint(
                model,
                sentence=sentence,
                prompt=prompt,
                answer=answer,
                dtype=dtype,
                device=device,
                quantize=quantize,
            )
        except Exception as exc:
            return {"error": str(exc)}

    @server.tool()
    def plan_program(path: str) -> Dict[str, Any]:
        """Resolve a portable weight program without loading a model."""
        from .program import load_program, program_preflight

        try:
            program = load_program(path)
            payload = program.to_dict()
            payload["preflight"] = program_preflight(program)
            return payload
        except Exception as exc:
            return {"error": str(exc)}

    @server.tool()
    def run_program(
        path: str,
        verify: Optional[bool] = None,
        out: Optional[str] = None,
        report_out: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a portable weight program with optional path overrides."""
        from .program import load_program
        from .program import run_program as execute_program

        try:
            result = execute_program(
                load_program(path),
                verify=verify,
                out_override=out,
                report_out=report_out,
            )
            return result.to_dict()
        except Exception as exc:
            return {"error": str(exc)}

    @server.tool()
    def verify_program_report(
        path: str,
        program_path: Optional[str] = None,
        check_model: bool = False,
        checkpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Verify a program report and optionally recheck its saved checkpoint."""
        from .program import verify_program_report as verify_report

        try:
            return verify_report(
                path,
                program_path=program_path,
                check_model=check_model,
                checkpoint=checkpoint,
            )
        except Exception as exc:
            return {"valid": False, "error": str(exc)}

    # -- model-level: one loaded session --------------------------------

    @server.tool()
    def open_model(path: str, dtype: str = "float32") -> Dict[str, Any]:
        """Load a model into the session. This is the expensive call."""
        from .session import Session

        session_box["session"] = Session.open(path, dtype=dtype)
        if current() is None:
            return NO_MODEL
        return current().state().__dict__

    @server.tool()
    def model_status() -> Dict[str, Any]:
        """Parameters, layers, edited tensors and budget usage."""
        if current() is None:
            return NO_MODEL
        return current().state().__dict__

    @server.tool()
    def capture_baseline(neutral_prompts: List[str],
                         corpus_texts: List[str] = ()) -> Dict[str, Any]:
        """Record reference distributions before editing anything."""
        if current() is None:
            return NO_MODEL
        return current().baseline(neutral_prompts, corpus_texts)

    @server.tool()
    def propose_edit(prompt: str, target: str, layer: int,
                     solver: str = "gradient", steps: int = 200,
                     max_rel_norm: Optional[float] = None) -> Dict[str, Any]:
        """Solve an edit and report its cost without writing it."""
        if current() is None:
            return NO_MODEL
        kwargs: Dict[str, Any] = {"solver": solver, "steps": steps}
        if max_rel_norm is not None:
            kwargs["max_rel_norm"] = max_rel_norm
        return current().propose(prompt, target, layer, **kwargs)

    @server.tool()
    def propose_batch(prompts: List[str], values: List[List[float]],
                      layer: int, preserve_prompts: List[str] = (),
                      max_rel_norm: Optional[float] = None,
                      clamp: bool = True, ridge: float = 0.0) -> Dict[str, Any]:
        """Solve many prompt/activation rows into one batch proposal."""
        if current() is None:
            return NO_MODEL
        return current().propose_batch(
            prompts, values, layer,
            preserve_prompts=preserve_prompts,
            max_rel_norm=max_rel_norm,
            clamp=clamp,
            ridge=ridge,
        )

    @server.tool()
    def propose_action(source: str, intent: str, layer: int,
                       module_suffix: str = "mlp.down_proj",
                       keys: Optional[List[List[float]]] = None,
                       values: Optional[List[List[float]]] = None,
                       preserve: Optional[List[List[float]]] = None,
                       preserve_prompts: List[str] = (),
                       max_rel_norm: Optional[float] = None,
                       timeout_seconds: float = 10.0) -> Dict[str, Any]:
        """Compile a constrained ``def edit(W): ...`` action into a pending delta.

        The source may use W, keys, values, preserve, max_rel_norm and the
        documented pure array helpers. Imports, filesystem access, loops,
        dunder attributes and arbitrary calls are rejected before execution.
        """
        if current() is None:
            return NO_MODEL
        kwargs: Dict[str, Any] = {
            "module_suffix": module_suffix,
            "keys": keys,
            "values": values,
            "preserve": preserve,
            "preserve_prompts": preserve_prompts,
            "timeout_seconds": timeout_seconds,
        }
        if max_rel_norm is not None:
            kwargs["max_rel_norm"] = max_rel_norm
        return current().propose_action(source, intent, layer, **kwargs)

    @server.tool()
    def apply_edit() -> Dict[str, Any]:
        """Write the pending proposal, charging the budget atomically."""
        if current() is None:
            return NO_MODEL
        return current().apply()

    @server.tool()
    def evaluate(prompts: List[str], targets: List[str]) -> Dict[str, Any]:
        """Accuracy on the targets, plus KL and perplexity against the baseline."""
        if current() is None:
            return NO_MODEL
        return current().evaluate(prompts, targets)

    @server.tool()
    def assess_wsp(prompts: List[str], targets: List[str],
                   headroom: Optional[Dict[str, float]] = None,
                   mechanism: Optional[Dict[str, Any]] = None,
                   seed: int = 0) -> Dict[str, Any]:
        """Evaluate the current edit plus a matched control and classify it."""
        if current() is None:
            return NO_MODEL
        return current().assess_wsp(
            prompts, targets, headroom=headroom, mechanism=mechanism, seed=seed
        )

    @server.tool()
    def assess_generation_wsp(prompts: List[str], expected: List[str],
                              max_new_tokens: int = 128,
                              headroom_max_new_tokens: Optional[int] = None,
                              match: str = "exact",
                              seed: int = 0) -> Dict[str, Any]:
        """Evaluate generation WSP with target, mechanism and matched-control evidence."""
        if current() is None:
            return NO_MODEL
        from .headroom import DecodingConfig

        headroom = (
            DecodingConfig(max_new_tokens=headroom_max_new_tokens)
            if headroom_max_new_tokens is not None else None
        )
        return current().assess_generation_wsp(
            prompts,
            expected,
            config=DecodingConfig(max_new_tokens=max_new_tokens),
            headroom_config=headroom,
            match=match,
            seed=seed,
        )

    @server.tool()
    def search_rome_layers(prompt: str, target: str, layers: List[int],
                           solver: str = "direction", max_rel_norm: Optional[float] = None,
                           metric: str = "accuracy",
                           max_evaluations: Optional[int] = None) -> Dict[str, Any]:
        """Try ROME proposals at each layer, then evaluate and roll back."""
        if current() is None:
            return NO_MODEL
        return current().search_rome_layers(
            prompt,
            target,
            layers,
            solver=solver,
            max_rel_norm=max_rel_norm,
            metric=metric,
            max_evaluations=max_evaluations,
        )

    @server.tool()
    def teach(sentence: Optional[str] = None,
              prompt: Optional[str] = None,
              answer: Optional[str] = None,
              layers: Optional[List[int]] = None,
              solver: str = "direction",
              max_rel_norm: float = 0.02,
              dry_run: bool = False) -> Dict[str, Any]:
        """Run the full teaching recipe: search, apply, verify, or roll back.

        The recipe chooses a depth window when ``layers`` is omitted, applies
        at most one tensor edit, and keeps it only when the target, budget,
        panel and KL checks pass.  Use ``rollback`` to undo an applied edit.
        """
        if current() is None:
            return NO_MODEL
        from .intent import parse_intent as compile_intent
        from .recipes import TeachPolicy, teach_fact

        if sentence:
            intent = compile_intent(sentence)
            prompt, answer = intent.prompt, intent.answer
        if not prompt or answer is None or answer == "":
            return {"error": "teach needs a sentence, or both prompt and answer"}
        result = teach_fact(
            current(),
            prompt,
            answer,
            layers=layers,
            policy=TeachPolicy(max_rel_norm=max_rel_norm),
            solver=solver,
            dry_run=dry_run,
        )
        return result.to_dict()

    @server.tool()
    def rollback() -> Dict[str, Any]:
        """Restore every edited tensor to its original bytes."""
        if current() is None:
            return NO_MODEL
        return current().rollback()

    return server


def main() -> None:
    create_server().run(transport="stdio")


if __name__ == "__main__":
    sys.exit(main())
