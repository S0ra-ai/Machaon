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
            "gate", "open_model", "model_status", "capture_baseline",
            "propose_edit", "apply_edit", "evaluate", "rollback",
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


if __name__ == "__main__":
    unittest.main()
