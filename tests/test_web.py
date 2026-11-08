"""The local web console is a third thin entry point over the same library."""
from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

try:
    from fastapi.testclient import TestClient

    from machaon.web.app import create_app
    from machaon.web.security import PathPolicy
    from machaon.web.service import WebService
    HAVE_WEB = True
except ImportError:
    TestClient = None
    create_app = None
    PathPolicy = None
    WebService = None
    HAVE_WEB = False


requires_web = unittest.skipUnless(HAVE_WEB, "fastapi + httpx not installed")


class FakeTools:
    def describe_checkpoint(self, path):
        return {
            "tensors": 2,
            "parameters": 42,
            "dtypes": {"F32": 42},
            "shards": 1,
            "path": path,
        }

    def list_tensors(self, path, contains=None, suffix=None, layer=None, limit=50):
        return [{"name": "model.layers.0.mlp.down_proj.weight",
                 "shape": [4, 3], "dtype": "F32"}]

    def audit_checkpoint_edit(self, base_path, edit_dir, max_rel_norm=None):
        return {"passed": True, "changed_tensors": 1}

    def apply_delta(self, base_path, tensor, delta_path, out_dir,
                    refuse_rescale=True):
        return {"tensor": tensor, "output": out_dir}

    def gate_metrics(self, metrics):
        missing = [key for key in ("rewrite_acc", "ppl_ratio")
                   if key not in metrics]
        return {
            "passed": not missing,
            "failures": [],
            "missing": missing,
            "n_gates": 2,
            "summary": "PASS" if not missing else "FAIL",
        }


class FakeSession:
    def __init__(self):
        self.model_id = "/models/toy"
        self._edited = []
        self._pending = False
        self._has_baseline = False
        self._proposal = None
        self.runtime = SimpleNamespace(read=lambda _name: [1.0, 2.0])

    def state(self):
        return SimpleNamespace(
            model="/models/toy",
            parameters=42,
            layers=3,
            edited=list(self._edited),
            budget_total=0.01 if self._edited else 0.0,
            budget_limit=0.10,
            has_baseline=self._has_baseline,
            quantize=None,
            has_pending=self._pending,
        )

    def baseline(self, prompts, corpus=()):
        self._has_baseline = True
        return {"prompts": len(prompts), "corpus": len(corpus)}

    def propose(self, **kwargs):
        self._pending = True
        self._proposal = {
            "module": "model.layers.1.mlp.down_proj.weight",
            "layer": kwargs["layer"],
            "relative_delta": 0.004,
            "target_probability": 0.98,
            "solver": kwargs["solver"],
            "steps": kwargs["steps"],
            "budget_after": 0.014,
        }
        return self._proposal

    def apply(self):
        self._pending = False
        self._edited.append("model.layers.1.mlp.down_proj.weight")
        return {"module": "model.layers.1.mlp.down_proj.weight",
                "relative_delta": 0.004, "budget_total": 0.014}

    def evaluate(self, prompts, targets):
        return {"accuracy": 1.0, "rewrite_acc": 1.0, "ppl_ratio": 1.001,
                "n": len(prompts)}

    def rollback(self):
        self._edited = []
        return {"restored": ["model.layers.1.mlp.down_proj.weight"]}

    def changes(self):
        return {"model.layers.1.mlp.down_proj.weight": [0.1, 0.2]}


def _factories(tmp):
    def session_factory(**kwargs):
        return FakeSession()

    return WebService(
        PathPolicy([tmp]),
        tools_module=FakeTools(),
        session_factory=session_factory,
    )


def _wait_for_job(client, job_id, headers):
    deadline = time.time() + 5
    while time.time() < deadline:
        response = client.get("/api/jobs/%s" % job_id, headers=headers)
        assert response.status_code == 200
        body = response.json()
        if body["status"] in ("done", "error"):
            return body
        time.sleep(0.01)
    raise AssertionError("job did not finish: %s" % job_id)


@requires_web
class TestWebApp(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "config.json").write_text("{}")
        self.token = "test-token"
        token = self.token
        service = _factories(self.root)
        self.app = create_app(
            token=token,
            allowed_roots=[self.root],
            service=service,
        )
        self.headers = {"X-Machaon-Token": token}
        self.client = TestClient(self.app)

    def tearDown(self):
        self.client.close()
        self.tmp.cleanup()

    def test_health_is_public_and_other_api_requires_token(self):
        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["auth"], "required")
        self.assertEqual(self.client.get("/api/models").status_code, 401)
        self.assertEqual(
            self.client.get("/api/models", headers=self.headers).status_code, 200
        )

    def test_path_policy_rejects_outside_paths(self):
        response = self.client.post(
            "/api/checkpoints/describe",
            headers=self.headers,
            json={"path": "/etc/passwd"},
        )
        self.assertEqual(response.status_code, 403)

    def test_describe_job_completes(self):
        response = self.client.post(
            "/api/checkpoints/describe",
            headers=self.headers,
            json={"path": str(self.root)},
        )
        self.assertEqual(response.status_code, 200)
        job = response.json()
        result = _wait_for_job(self.client, job["job"], self.headers)
        self.assertEqual(result["status"], "done")
        self.assertEqual(result["result"]["tensors"], 2)

    def test_job_events_are_streamable(self):
        response = self.client.post(
            "/api/checkpoints/describe",
            headers=self.headers,
            json={"path": str(self.root)},
        )
        job_id = response.json()["job"]
        _wait_for_job(self.client, job_id, self.headers)
        stream = self.client.get(
            "/api/jobs/%s/events?token=%s" % (job_id, self.token)
        )
        self.assertEqual(stream.status_code, 200)
        self.assertIn("data:", stream.text)
        self.assertIn("event: end", stream.text)
        payloads = [
            json.loads(line[len("data: "):])
            for line in stream.text.splitlines()
            if line.startswith("data: ")
        ]
        self.assertTrue(any(event["type"] == "result" for event in payloads))

    def test_session_edit_lifecycle(self):
        opened = self.client.post(
            "/api/session/open",
            headers=self.headers,
            json={"path": str(self.root), "dtype": "auto"},
        )
        job = _wait_for_job(self.client, opened.json()["job"], self.headers)
        self.assertEqual(job["result"]["model"], "/models/toy")

        baseline = self.client.post(
            "/api/session/baseline",
            headers=self.headers,
            json={"neutral_prompts": ["a", "b"]},
        )
        _wait_for_job(self.client, baseline.json()["job"], self.headers)

        proposed = self.client.post(
            "/api/edit/propose",
            headers=self.headers,
            json={"prompt": "a", "target": " b", "layer": 1},
        )
        proposed_job = _wait_for_job(
            self.client, proposed.json()["job"], self.headers
        )
        self.assertEqual(proposed_job["result"]["proposal"]["layer"], 1)

        applied = self.client.post("/api/edit/apply", headers=self.headers, json={})
        applied_job = _wait_for_job(self.client, applied.json()["job"], self.headers)
        self.assertEqual(applied_job["status"], "done")
        state = self.client.get("/api/session/state", headers=self.headers).json()
        self.assertTrue(state["edited"])
        self.assertFalse(state["has_pending"])

        evaluated = self.client.post(
            "/api/edit/evaluate",
            headers=self.headers,
            json={"prompts": ["a"], "targets": [" b"]},
        )
        evaluation = _wait_for_job(
            self.client, evaluated.json()["job"], self.headers
        )
        self.assertTrue(evaluation["result"]["gate"]["passed"])

    def test_model_discovery_lists_marker_directory(self):
        response = self.client.get("/api/models", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            Path(response.json()["models"][0]["path"]).resolve(),
            self.root.resolve(),
        )

    def test_unknown_job_is_404(self):
        response = self.client.get(
            "/api/jobs/not-a-real-job", headers=self.headers
        )
        self.assertEqual(response.status_code, 404)

    def test_gate_endpoint_is_fail_closed(self):
        response = self.client.post(
            "/api/gates/check",
            headers=self.headers,
            json={"metrics": {"rewrite_acc": 1.0}},
        )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["passed"])
        self.assertIn("ppl_ratio", response.json()["missing"])

    def test_save_checkpoint_uses_the_checkpoint_writer(self):
        opened = self.client.post(
            "/api/session/open",
            headers=self.headers,
            json={"path": str(self.root), "dtype": "auto"},
        )
        _wait_for_job(self.client, opened.json()["job"], self.headers)
        applied = self.client.post("/api/edit/apply", headers=self.headers, json={})
        _wait_for_job(self.client, applied.json()["job"], self.headers)

        class FakeCheckpoint:
            def __init__(self, path):
                self.path = path

            def write_tensor(self, name, values, out_dir):
                return {"tensor": name, "shard": "shard-1", "rewritten": "shard-1"}

        with mock.patch("machaon.adapters.Checkpoint", FakeCheckpoint):
            response = self.client.post(
                "/api/session/save",
                headers=self.headers,
                json={"out_dir": str(self.root / "edited")},
            )
            result = _wait_for_job(
                self.client, response.json()["job"], self.headers
            )
        self.assertEqual(result["status"], "done")
        self.assertEqual(
            Path(result["result"]["output"]).resolve(),
            (self.root / "edited").resolve(),
        )
        self.assertEqual(result["result"]["edited"],
                         ["model.layers.1.mlp.down_proj.weight"])


if __name__ == "__main__":
    unittest.main()
