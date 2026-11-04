"""End-to-end MCP smoke test: start the server, list tools, call one.

    python scripts/mcp_smoke.py

Needs the optional extra (``pip install "machaon[mcp]"``). Exits non-zero if the
tool surface changes or a call fails.
"""
from __future__ import annotations

import asyncio
import json
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

EXPECTED = {
    "describe_checkpoint", "list_tensors", "audit_edit", "apply_delta", "gate",
    "open_model", "model_status", "capture_baseline", "propose_edit",
    "apply_edit", "evaluate", "rollback",
}


async def main() -> int:
    metrics = {
        "rewrite_acc": 1.0, "paraphrase_acc": 0.8, "unrelated_acc": 1.0,
        "kl_p95": 0.01, "kl_max": 0.1, "mmlu_delta": 0.0,
        "ppl_ratio": 1.0, "longform_contradictions": 0,
    }
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "machaon.mcp_server"],
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            listed = await session.list_tools()
            names = {tool.name for tool in listed.tools}
            print("tools:", len(names))
            if names != EXPECTED:
                print("unexpected tool set:", sorted(names ^ EXPECTED))
                return 1

            passed = await session.call_tool("gate", {"metrics": metrics})
            verdict = json.loads(passed.content[0].text)
            print("gate on good metrics: passed =", verdict["passed"])
            if not verdict["passed"]:
                return 1

            failed = await session.call_tool("gate", {"metrics": {}})
            verdict = json.loads(failed.content[0].text)
            print("gate on empty metrics: passed =", verdict["passed"])
            if verdict["passed"]:
                return 1

            status = await session.call_tool("model_status", {})
            text = status.content[0].text
            print("model_status without a model:", text[:60])
            if "no model is open" not in text:
                return 1

    print("MCP smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
