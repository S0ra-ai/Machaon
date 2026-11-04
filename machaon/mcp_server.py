"""MCP stdio server exposing the file-level operations.

    python -m machaon.mcp_server

The transport is deliberately thin: every tool calls a function in
:mod:`machaon.tools`. Nothing here loads a model, so the server is cheap to run
alongside an agent and safe to restart.

Requires the optional extra: ``pip install "machaon[mcp]"``.
"""
from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional


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
