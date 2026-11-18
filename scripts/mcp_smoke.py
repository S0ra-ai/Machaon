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
    "wsp_audit", "wsp_save", "verify_binding", "plan_surgery",
    "diagnose_failure", "parse_intent", "check_saved",
    "open_model", "model_status", "capture_baseline", "propose_edit",
    "propose_batch", "propose_action", "apply_edit", "evaluate",
    "assess_wsp", "assess_generation_wsp", "search_rome_layers", "teach",
    "rollback",
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

            with open("examples/wsp_report.json", encoding="utf-8") as handle:
                report = json.load(handle)
            classified = await session.call_tool("wsp_audit", {"report": report})
            verdict = json.loads(classified.content[0].text)
            print("wsp_audit label:", verdict["label"])
            if not verdict["accepted"] or verdict["label"] != "repair":
                return 1

            planned = await session.call_tool("plan_surgery", {
                "failure_type": "obstruction",
                "layer_scores": {"12": 0.1, "14": 0.3},
            })
            plan = json.loads(planned.content[0].text)
            print("plan_surgery route/layer:", plan["route"], plan["layer"])
            if plan["route"] != "edit" or plan["layer"] != 14:
                return 1

            diagnosis = await session.call_tool("diagnose_failure", {
                "base": {"accuracy": 0.2},
                "headroom": {"accuracy": 0.9},
                "mechanism": {
                    "base": {"loop_rate": 0.9},
                    "headroom": {"loop_rate": 0.3},
                },
            })
            result = json.loads(diagnosis.content[0].text)
            print("diagnose_failure:", result["failure_type"])
            if result["failure_type"] != "obstruction":
                return 1

            checked = await session.call_tool("check_saved", {
                "model": "/tmp/definitely-not-a-model",
                "sentence": "please make it better",
            })
            text = checked.content[0].text
            print("check_saved invalid sentence:", text[:60])
            if "could not read" not in text and "error" not in text:
                return 1

            status = await session.call_tool("model_status", {})
            text = status.content[0].text
            print("model_status without a model:", text[:60])
            if "no model is open" not in text:
                return 1

            action = await session.call_tool("propose_action", {
                "source": "def edit(W):\n    return 0.001 * W\n",
                "intent": "smoke-test action",
                "layer": 0,
            })
            text = action.content[0].text
            print("propose_action without a model:", text[:60])
            if "no model is open" not in text:
                return 1

            generation_wsp = await session.call_tool("assess_generation_wsp", {
                "prompts": ["p"],
                "expected": ["a"],
            })
            text = generation_wsp.content[0].text
            print("assess_generation_wsp without a model:", text[:60])
            if "no model is open" not in text:
                return 1

            search = await session.call_tool("search_rome_layers", {
                "prompt": "p",
                "target": "a",
                "layers": [0, 1],
            })
            text = search.content[0].text
            print("search_rome_layers without a model:", text[:60])
            if "no model is open" not in text:
                return 1

    print("MCP smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
