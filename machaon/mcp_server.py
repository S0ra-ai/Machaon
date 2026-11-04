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


def create_server():
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            'the MCP server needs the optional extra: pip install "machaon[mcp]"'
        ) from exc

    from . import tools

    server = FastMCP("Machaon")

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

    return server


def main() -> None:
    create_server().run(transport="stdio")


if __name__ == "__main__":
    sys.exit(main())
