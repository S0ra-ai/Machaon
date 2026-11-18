"""The MCP wiring, without the mcp SDK.

A recording stand-in for FastMCP captures every decorated tool. That checks the
thing that actually breaks: a tool that is not registered, is registered under
the wrong name, or does not delegate to `machaon.tools` / `machaon.session`.
"""
import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

FIXTURE = Path(__file__).parent / "fixtures" / "openwsp_v1"

try:
    import torch
    from safetensors.torch import save_file
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

requires_torch = unittest.skipUnless(HAVE_TORCH, "torch + safetensors not installed")


class FakeFastMCP:
    def __init__(self, name):
        self.name = name
        self.tools = {}

    def tool(self):
        def decorator(fn):
            self.tools[fn.__name__] = fn
            return fn

        return decorator

    def run(self, transport=None):
        self.transport = transport


class FakeMCPModules:
    def __enter__(self):
        self.saved = {k: sys.modules.get(k) for k in
                      ("mcp", "mcp.server", "mcp.server.fastmcp")}
        mcp = types.ModuleType("mcp")
        server = types.ModuleType("mcp.server")
        fastmcp = types.ModuleType("mcp.server.fastmcp")
        fastmcp.FastMCP = FakeFastMCP
        sys.modules["mcp"] = mcp
        sys.modules["mcp.server"] = server
        sys.modules["mcp.server.fastmcp"] = fastmcp

    def __exit__(self, *exc):
        for name, module in self.saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        return False


def build_server():
    import machaon.mcp_server as server_module

    importlib.reload(server_module)
    return server_module.create_server()


class TestRegistration(unittest.TestCase):
    def test_every_documented_tool_is_registered(self):
        with FakeMCPModules():
            server = build_server()
        expected = {
            "describe_checkpoint", "list_tensors", "audit_edit", "apply_delta",
            "gate", "wsp_audit", "wsp_save", "verify_binding", "plan_surgery",
            "diagnose_failure", "parse_intent", "check_saved",
            "plan_program", "run_program",
            "open_model", "model_status", "capture_baseline",
            "propose_edit", "propose_batch", "propose_action", "apply_edit",
            "evaluate", "assess_wsp", "assess_generation_wsp",
            "search_rome_layers", "teach", "rollback",
        }
        self.assertEqual(set(server.tools), expected)

    def test_tools_have_docstrings(self):
        with FakeMCPModules():
            server = build_server()
        for name, fn in server.tools.items():
            self.assertTrue((fn.__doc__ or "").strip(), "%s has no docstring" % name)

    def test_session_tools_refuse_without_a_model(self):
        with FakeMCPModules():
            server = build_server()
        for name in ("model_status", "apply_edit", "rollback"):
            self.assertIn("no model is open", server.tools[name]()["error"], msg=name)
        refusal = server.tools["propose_action"](
            "def edit(W):\n    return 0.001 * W\n", "test", 0
        )
        self.assertIn("no model is open", refusal["error"])
        refusal = server.tools["assess_generation_wsp"](["p"], ["a"])
        self.assertIn("no model is open", refusal["error"])
        refusal = server.tools["search_rome_layers"]("p", "a", [0, 1])
        self.assertIn("no model is open", refusal["error"])
        refusal = server.tools["teach"]("The capital is -> Paris")
        self.assertIn("no model is open", refusal["error"])

    def test_parse_intent_delegates(self):
        with FakeMCPModules():
            server = build_server()
        payload = server.tools["parse_intent"](
            "教会模型：北京的首都是 → 北京"
        )
        self.assertEqual(payload["kind"], "teach")
        self.assertEqual(payload["prompt"], "北京的首都是")
        self.assertEqual(payload["answer"], "北京")

    def test_program_tools_parse_without_a_model_and_report_errors(self):
        with FakeMCPModules():
            server = build_server()
        with tempfile.NamedTemporaryFile(
            "w", suffix=".machaon", delete=False
        ) as handle:
            handle.write(
                "model: /tmp/definitely-not-a-model\n"
                "output: /tmp/definitely-not-an-output\n"
                "teach: The capital of France is -> Paris\n"
            )
            path = handle.name
        try:
            plan = server.tools["plan_program"](path)
            self.assertEqual(len(plan["facts"]), 1)
            self.assertEqual(plan["facts"][0]["prompt"], "The capital of France is")
            failed = server.tools["run_program"](path, verify=False)
            self.assertIn("error", failed)
        finally:
            Path(path).unlink(missing_ok=True)


@requires_torch
class TestDelegation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        originals = {
            p.name[: -len(".npy")]: np.load(p)
            for p in (FIXTURE / "originals").glob("*.npy")
        }
        self.model = self.dir / "model.safetensors"
        save_file({k: torch.from_numpy(v) for k, v in originals.items()}, str(self.model))

    def tearDown(self):
        self.tmp.cleanup()

    def test_describe_checkpoint_delegates(self):
        with FakeMCPModules():
            server = build_server()
        summary = server.tools["describe_checkpoint"](str(self.model))
        self.assertEqual(summary["tensors"], 2)

    def test_audit_edit_delegates(self):
        with FakeMCPModules():
            server = build_server()
        report = server.tools["audit_edit"](str(self.model), str(FIXTURE))
        self.assertTrue(report["passed"])
        self.assertEqual(report["changed_tensors"], 2)

    def test_gate_delegates(self):
        with FakeMCPModules():
            server = build_server()
        self.assertFalse(server.tools["gate"]({})["passed"])

    def test_plan_delegates(self):
        with FakeMCPModules():
            server = build_server()
        plan = server.tools["plan_surgery"](
            "obstruction", layer_scores={2: 0.4}
        )
        self.assertEqual(plan["route"], "edit")
        self.assertEqual(plan["layer"], 2)

    def test_diagnose_delegates(self):
        with FakeMCPModules():
            server = build_server()
        diagnosis = server.tools["diagnose_failure"](
            {"accuracy": 0.2},
            headroom={"accuracy": 0.9},
            mechanism={
                "base": {"loop_rate": 0.9},
                "headroom": {"loop_rate": 0.3},
            },
        )
        self.assertEqual(diagnosis["failure_type"], "obstruction")


if __name__ == "__main__":
    unittest.main()
