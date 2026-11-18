"""The local web console is a third thin entry point over the same library."""
from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
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

    def wsp_audit(self, report):
        return {
            "accepted": True,
            "label": "repair",
            "reason": "test repair",
            "classification": {"control_gap": 0.76, "headroom_gap": 0.02},
        }

    def save_wsp_evidence(self, out_dir, report, metadata=None):
        return {
            "path": out_dir,
            "format": "machaon.wsp.evidence",
            "format_version": 1,
            "evidence_sha256": "abc123",
            "label": report.get("label", "repair"),
            "accepted": True,
        }

    def verify_wsp_binding(self, edit_dir, evidence_dir):
        return {
            "passed": True,
            "edit_sha256": "edit123",
            "evidence_sha256": "evidence123",
            "label": "repair",
            "errors": [],
        }

class FakeSession:
    def __init__(self):
        self.model_id = "/models/toy"
        self._edited = []
        self._pending = False
        self._has_baseline = False
        self._proposal = None
        self.densified = []
        self.runtime = SimpleNamespace(
            read=lambda _name: [1.0, 2.0],
            densify_from=lambda layer: self.densified.append(layer) or 2,
        )

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

    def propose_batch(self, **kwargs):
        self._pending = True
        return {
            "module": "model.layers.1.mlp.down_proj.weight",
            "layer": kwargs["layer"],
            "relative_delta": 0.003,
            "target_error": 1e-12,
            "per_key_error": [0.0] * len(kwargs["prompts"]),
            "exact": True,
            "clamped": False,
            "rank": 2,
            "budget_after": 0.013,
        }

    def apply(self):
        self._pending = False
        self._edited.append("model.layers.1.mlp.down_proj.weight")
        return {"module": "model.layers.1.mlp.down_proj.weight",
                "relative_delta": 0.004, "budget_total": 0.014}

    def evaluate(self, prompts, targets):
        return {"accuracy": 1.0, "rewrite_acc": 1.0, "ppl_ratio": 1.001,
                "n": len(prompts)}

    def assess_wsp(self, prompts, targets, headroom=None, seed=0,
                   target_metric="accuracy", mechanism=None):
        result = {
            "accepted": True,
            "label": "repair",
            "reason": "session repair",
            "classification": {"control_gap": 0.7, "headroom_gap": 0.03},
            "protocol": {"target_metric": target_metric},
            "candidate_gate": {"passed": False},
            "control_gates": {},
        }
        if mechanism is not None:
            result["mechanism"] = {
                "passed": True,
                "label": "obstruction_relieved",
            }
        return result

    def rollback(self):
        self._edited = []
        return {"restored": ["model.layers.1.mlp.down_proj.weight"]}

    def changes(self):
        return {"model.layers.1.mlp.down_proj.weight": [0.1, 0.2]}

    def export_openwsp(self, path, base_model=None, revision=None, metadata=None):
        return {
            "format": "openwsp.edit",
            "format_version": 1,
            "edit_sha256": "edit123",
            "base_model": base_model,
            "machaon": metadata or {},
        }


class FakeFetch:
    RECOMMENDED = {
        "smol135m": {"repo": "fake/small", "name": "Smol"},
        "qwen05b": {"repo": "fake/qwen", "name": "Qwen"},
    }

    def fetch_model(self, repo, dest, endpoint=None, require_new=False,
                    progress=None):
        path = Path(dest)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text("{}")
        if progress:
            progress(phase="file", file="model.safetensors",
                     index=1, count=2, cached=False)
            progress(phase="download", file="model.safetensors",
                     bytes=5, total=10)
            progress(phase="done", dest=str(path))
        return {
            "repo": repo,
            "dest": str(path),
            "endpoint": endpoint or "huggingface",
            "files": ["config.json", "model.safetensors"],
            "downloaded": ["config.json", "model.safetensors"],
            "cached": [],
            "tensors": 1,
            "parameters": 10,
            "shards": 1,
        }


def many_recipe(factory, base_model, facts, out_dir, **kwargs):
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "final").mkdir(exist_ok=True)
    (path / "report.json").write_text("{}")
    return {
        "status": "applied",
        "accepted": True,
        "base_model": base_model,
        "out_dir": str(path),
        "final": str(path / "final"),
        "facts": len(facts),
        "budget": kwargs.get("budget", 0.10),
        "cumulative_relative_delta": 0.02,
        "panel_size": 8,
        "global_panel_agreement": 1.0,
        "global_kl_p95": 0.0,
        "global_passed": True,
        "stopped_at": None,
        "reason": "",
        "summary": "test chain applied",
        "steps": [],
        "report_path": str(path / "report.json"),
    }


def program_runner(program):
    return {
        "program": program.to_dict(),
        "run": {
            "status": "applied",
            "out_dir": program.output,
            "final": str(Path(program.output) / "final"),
            "facts": len(program.facts),
        },
        "verification": {
            "passed": True,
            "facts": [
                {
                    "index": index,
                    "prompt": fact["prompt"],
                    "answer": fact["answer"],
                    "passed": True,
                }
                for index, fact in enumerate(program.facts, 1)
            ],
        },
        "status": "applied",
        "accepted": True,
        "summary": "test program applied",
        "report_path": str(Path(program.output) / "program-report.json"),
        "model_fetched": False,
        "effective_output": program.output,
    }


def _factories(tmp):
    def session_factory(**kwargs):
        return FakeSession()

    def recipe(session, prompt, answer, **kwargs):
        session.apply()
        return SimpleNamespace(
            status="applied",
            to_dict=lambda: {
                "status": "applied",
                "accepted": True,
                "prompt": prompt,
                "answer": answer,
                "answer_text": answer,
                "best_layer": 1,
                "tensor": "model.layers.1.mlp.down_proj.weight",
                "relative_delta": 0.004,
                "checks": [{"name": "target", "passed": True}],
                "warnings": [],
                "summary": "test edit applied",
            },
        )

    return WebService(
        PathPolicy([tmp]),
        tools_module=FakeTools(),
        session_factory=session_factory,
        recipe=recipe,
        fetch_module=FakeFetch(),
        many_recipe=many_recipe,
        program_runner=program_runner,
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
        self.assertIn("torch", health.json()["capabilities"])
        self.assertIn("transformers", health.json()["capabilities"])
        self.assertEqual(self.client.get("/api/models").status_code, 401)
        self.assertEqual(
            self.client.get("/api/models", headers=self.headers).status_code, 200
        )

    def test_capability_probe_reports_missing_runtimes(self):
        with mock.patch("importlib.util.find_spec", return_value=None):
            caps = WebService.capabilities()
        self.assertEqual(
            caps, {"torch": False, "transformers": False, "safetensors": False}
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
            json={"prompt": "a", "target": " b", "layer": 1,
                  "densify_from_layer": True},
        )
        proposed_job = _wait_for_job(
            self.client, proposed.json()["job"], self.headers
        )
        self.assertEqual(proposed_job["result"]["proposal"]["layer"], 1)
        self.assertEqual(proposed_job["result"]["densified"], 2)

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

    def test_check_endpoint_evaluates_a_saved_fact(self):
        response = self.client.post(
            "/api/check",
            headers=self.headers,
            json={
                "path": str(self.root),
                "sentence": "The capital of France is -> Paris",
            },
        )
        job = _wait_for_job(self.client, response.json()["job"], self.headers)
        self.assertEqual(job["status"], "done")
        self.assertTrue(job["result"]["passed"])
        self.assertEqual(job["result"]["accuracy"], 1.0)
        self.assertEqual(job["result"]["answer_text"], " Paris")

    def test_check_endpoint_evaluates_a_fact_list(self):
        response = self.client.post(
            "/api/check",
            headers=self.headers,
            json={
                "path": str(self.root),
                "facts": [
                    "The capital of France is -> Paris",
                    "The city is -> Rome",
                ],
            },
        )
        job = _wait_for_job(self.client, response.json()["job"], self.headers)
        self.assertEqual(job["status"], "done")
        self.assertTrue(job["result"]["passed"])
        self.assertEqual(len(job["result"]["facts"]), 2)
        self.assertEqual(job["result"]["facts"][1]["index"], 2)

    def test_program_plan_resolves_a_pasted_contract(self):
        text = (
            "version: 1\n"
            "model: ./base\n"
            "output: ./edited\n"
            "teach: The capital of France is -> Paris\n"
            "protect: Two plus two equals\n"
        )
        response = self.client.post(
            "/api/program/plan",
            headers=self.headers,
            json={"program": text, "base_dir": str(self.root)},
        )
        self.assertEqual(response.status_code, 200)
        plan = response.json()
        self.assertEqual(plan["format"], "machaon.program/1")
        self.assertEqual(len(plan["facts"]), 1)
        self.assertEqual(plan["facts"][0]["prompt"], "The capital of France is")
        self.assertEqual(plan["base_dir"], str(self.root.resolve()))

    def test_program_plan_refuses_paths_outside_roots(self):
        response = self.client.post(
            "/api/program/plan",
            headers=self.headers,
            json={
                "program": (
                    "model: /etc/base\noutput: /etc/edited\n"
                    "teach: The capital is -> Paris\n"
                ),
                "base_dir": str(self.root),
            },
        )
        self.assertIn(response.status_code, (400, 403))

    def test_program_run_completes_as_a_job(self):
        text = (
            "model: ./base\noutput: ./edited\n"
            "teach: The capital of France is -> Paris\n"
        )
        response = self.client.post(
            "/api/program/run",
            headers=self.headers,
            json={"program": text, "base_dir": str(self.root)},
        )
        self.assertEqual(response.status_code, 200)
        job = _wait_for_job(self.client, response.json()["job"], self.headers)
        self.assertEqual(job["status"], "done")
        self.assertTrue(job["result"]["accepted"])
        self.assertTrue(job["result"]["verification"]["passed"])

    def test_fetch_starter_downloads_below_an_allowed_root(self):
        dest = self.root / "Smol"
        response = self.client.post(
            "/api/models/fetch",
            headers=self.headers,
            json={"starter": "smol135m", "dest": str(dest)},
        )
        self.assertEqual(response.status_code, 200)
        job_id = response.json()["job"]
        job = _wait_for_job(self.client, job_id, self.headers)
        self.assertEqual(job["status"], "done")
        self.assertEqual(job["result"]["starter"], "smol135m")
        self.assertEqual(job["result"]["dest"], str(dest.resolve()))
        self.assertTrue((dest / "config.json").exists())
        events = self.client.get(
            "/api/jobs/%s/events?token=%s" % (job_id, self.token)
        )
        self.assertIn('"type": "progress"', events.text)
        self.assertIn('"phase": "download"', events.text)

    def test_job_registry_emits_progress_records(self):
        from machaon.web.jobs import JobRegistry

        jobs = JobRegistry()
        try:
            job = jobs.start(
                "probe",
                lambda progress: progress(
                    {"phase": "download", "bytes": 5, "total": 10}
                ),
                with_progress=True,
            )
            self.assertTrue(job.wait(5))
            events = job.events_after(0)
            progress = [event for event in events if event["type"] == "progress"]
            self.assertEqual(progress[0]["data"]["phase"], "download")
            self.assertEqual(progress[0]["data"]["bytes"], 5)
        finally:
            jobs.shutdown(wait=False)

    def test_fetch_unknown_starter_is_a_job_error(self):
        response = self.client.post(
            "/api/models/fetch",
            headers=self.headers,
            json={"starter": "nope", "dest": str(self.root / "x")},
        )
        job = _wait_for_job(self.client, response.json()["job"], self.headers)
        self.assertEqual(job["status"], "error")
        self.assertIn("unknown starter", job["error"])

    def test_fetch_resumes_an_existing_directory(self):
        dest = self.root / "already"
        dest.mkdir()
        (dest / "model.safetensors.part").write_bytes(b"partial")
        response = self.client.post(
            "/api/models/fetch",
            headers=self.headers,
            json={"starter": "smol135m", "dest": str(dest)},
        )
        job = _wait_for_job(self.client, response.json()["job"], self.headers)
        self.assertEqual(job["status"], "done")
        self.assertEqual(job["result"]["starter"], "smol135m")

    def test_fetch_path_outside_the_roots_is_refused(self):
        response = self.client.post(
            "/api/models/fetch",
            headers=self.headers,
            json={"starter": "smol135m", "dest": "/etc/machaon-model"},
        )
        self.assertEqual(response.status_code, 403)

    def test_teach_many_runs_on_the_open_session(self):
        opened = self.client.post(
            "/api/session/open",
            headers=self.headers,
            json={"path": str(self.root), "dtype": "auto"},
        )
        _wait_for_job(self.client, opened.json()["job"], self.headers)
        response = self.client.post(
            "/api/teach_many",
            headers=self.headers,
            json={
                "facts": [
                    "The capital is -> Paris",
                    "The city is -> Rome",
                ],
                "out_dir": str(self.root / "many"),
            },
        )
        job = _wait_for_job(self.client, response.json()["job"], self.headers)
        self.assertEqual(job["status"], "done")
        result = job["result"]["result"]
        self.assertEqual(result["status"], "applied")
        self.assertEqual(result["facts"], 2)
        self.assertTrue((self.root / "many" / "final").exists())
        self.assertTrue(job["result"]["state"]["ready"])

    def test_teach_many_requires_an_open_session(self):
        response = self.client.post(
            "/api/teach_many",
            headers=self.headers,
            json={
                "facts": ["The capital is -> Paris"],
                "out_dir": str(self.root / "many"),
            },
        )
        job = _wait_for_job(self.client, response.json()["job"], self.headers)
        self.assertEqual(job["status"], "error")
        self.assertIn("no model is open", job["error"])

    def test_teach_many_refuses_an_existing_output(self):
        opened = self.client.post(
            "/api/session/open",
            headers=self.headers,
            json={"path": str(self.root), "dtype": "auto"},
        )
        _wait_for_job(self.client, opened.json()["job"], self.headers)
        out = self.root / "many-exists"
        out.mkdir()
        response = self.client.post(
            "/api/teach_many",
            headers=self.headers,
            json={"facts": ["The capital is -> Paris"], "out_dir": str(out)},
        )
        job = _wait_for_job(self.client, response.json()["job"], self.headers)
        self.assertEqual(job["status"], "error")
        self.assertIn("already exists", job["error"])

    def test_teach_many_path_outside_the_roots_is_refused(self):
        response = self.client.post(
            "/api/teach_many",
            headers=self.headers,
            json={
                "facts": ["The capital is -> Paris"],
                "out_dir": "/etc/machaon-many",
            },
        )
        self.assertEqual(response.status_code, 403)

    def test_teach_preview_compiles_a_sentence_without_a_session(self):
        response = self.client.post(
            "/api/teach/preview",
            headers=self.headers,
            json={"sentence": "The Eiffel Tower is in the city of -> Rome"},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["prompt"], "The Eiffel Tower is in the city of")
        self.assertEqual(payload["answer"], "Rome")
        self.assertIn("command", payload)

    def test_teach_preview_rejects_unstructured_text(self):
        response = self.client.post(
            "/api/teach/preview",
            headers=self.headers,
            json={"sentence": "please make it better"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Try one of", response.json()["detail"])

    def test_teach_requires_an_open_session(self):
        response = self.client.post(
            "/api/teach",
            headers=self.headers,
            json={"sentence": "The capital is -> Paris"},
        )
        job = _wait_for_job(self.client, response.json()["job"], self.headers)
        self.assertEqual(job["status"], "error")
        self.assertIn("no model is open", job["error"])

    def test_teach_runs_the_recipe_on_the_open_session(self):
        opened = self.client.post(
            "/api/session/open",
            headers=self.headers,
            json={"path": str(self.root), "dtype": "auto"},
        )
        _wait_for_job(self.client, opened.json()["job"], self.headers)
        response = self.client.post(
            "/api/teach",
            headers=self.headers,
            json={"sentence": "The capital is -> Paris"},
        )
        job = _wait_for_job(self.client, response.json()["job"], self.headers)
        self.assertEqual(job["status"], "done")
        result = job["result"]
        self.assertEqual(result["result"]["status"], "applied")
        self.assertEqual(result["intent"]["prompt"], "The capital is")
        self.assertTrue(result["state"]["edited"])


    def test_batch_propose_endpoint(self):
        opened = self.client.post(
            "/api/session/open",
            headers=self.headers,
            json={"path": str(self.root), "dtype": "auto"},
        )
        _wait_for_job(self.client, opened.json()["job"], self.headers)

        response = self.client.post(
            "/api/edit/propose_batch",
            headers=self.headers,
            json={
                "prompts": ["a", "b"],
                "values": [[0.0, 0.0], [0.0, 0.0]],
                "layer": 1,
            },
        )
        self.assertEqual(response.status_code, 200)
        job = _wait_for_job(self.client, response.json()["job"], self.headers)
        self.assertEqual(job["status"], "done")
        self.assertTrue(job["result"]["proposal"]["exact"])
        self.assertEqual(job["result"]["proposal"]["rank"], 2)

    def test_wsp_audit_endpoint(self):
        response = self.client.post(
            "/api/wsp/audit",
            headers=self.headers,
            json={"report": {"base": {}, "candidate": {}}},
        )
        self.assertEqual(response.status_code, 200)
        job = _wait_for_job(self.client, response.json()["job"], self.headers)
        self.assertEqual(job["status"], "done")
        self.assertEqual(job["result"]["label"], "repair")
        self.assertTrue(job["result"]["accepted"])

    def test_wsp_save_endpoint(self):
        response = self.client.post(
            "/api/wsp/save",
            headers=self.headers,
            json={
                "report": {"label": "repair", "accepted": True},
                "out_dir": str(self.root / "evidence"),
            },
        )
        self.assertEqual(response.status_code, 200)
        job = _wait_for_job(self.client, response.json()["job"], self.headers)
        self.assertEqual(job["status"], "done")
        self.assertEqual(job["result"]["evidence_sha256"], "abc123")

    def test_wsp_session_endpoint(self):
        opened = self.client.post(
            "/api/session/open",
            headers=self.headers,
            json={"path": str(self.root), "dtype": "auto"},
        )
        _wait_for_job(self.client, opened.json()["job"], self.headers)
        response = self.client.post(
            "/api/wsp/session",
            headers=self.headers,
            json={
                "prompts": ["a"],
                "targets": ["b"],
                "headroom": {"accuracy": 0.9},
            },
        )
        self.assertEqual(response.status_code, 200)
        job = _wait_for_job(self.client, response.json()["job"], self.headers)
        self.assertEqual(job["status"], "done")
        self.assertEqual(job["result"]["label"], "repair")

    def test_wsp_session_evidence_endpoint(self):
        opened = self.client.post(
            "/api/session/open",
            headers=self.headers,
            json={"path": str(self.root), "dtype": "auto"},
        )
        _wait_for_job(self.client, opened.json()["job"], self.headers)
        response = self.client.post(
            "/api/wsp/session_evidence",
            headers=self.headers,
            json={
                "prompts": ["a"],
                "targets": ["b"],
                "headroom": {"accuracy": 0.9},
                "out_dir": str(self.root / "evidence"),
            },
        )
        self.assertEqual(response.status_code, 200)
        job = _wait_for_job(self.client, response.json()["job"], self.headers)
        self.assertEqual(job["status"], "done")
        self.assertEqual(job["result"]["report"]["label"], "repair")
        self.assertEqual(job["result"]["evidence"]["evidence_sha256"], "abc123")

    def test_wsp_session_mechanism_is_forwarded(self):
        opened = self.client.post(
            "/api/session/open",
            headers=self.headers,
            json={"path": str(self.root), "dtype": "auto"},
        )
        _wait_for_job(self.client, opened.json()["job"], self.headers)
        response = self.client.post(
            "/api/wsp/session",
            headers=self.headers,
            json={
                "prompts": ["a"],
                "targets": ["b"],
                "mechanism": {
                    "base": {"loop_rate": 0.9},
                    "candidate": {"loop_rate": 0.3},
                    "controls": {"random": {"loop_rate": 0.85}},
                    "protocol": {
                        "metrics": ["loop_rate"],
                        "directions": {"loop_rate": "lower"},
                    },
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        job = _wait_for_job(self.client, response.json()["job"], self.headers)
        self.assertEqual(job["status"], "done")
        self.assertEqual(
            job["result"]["mechanism"]["label"], "obstruction_relieved"
        )

    def test_binding_verify_endpoint(self):
        (self.root / "edit").mkdir()
        (self.root / "evidence").mkdir()
        response = self.client.post(
            "/api/binding/verify",
            headers=self.headers,
            json={
                "edit_dir": str(self.root / "edit"),
                "evidence_dir": str(self.root / "evidence"),
            },
        )
        self.assertEqual(response.status_code, 200)
        job = _wait_for_job(self.client, response.json()["job"], self.headers)
        self.assertEqual(job["status"], "done")
        self.assertTrue(job["result"]["passed"])
        self.assertEqual(job["result"]["evidence_sha256"], "evidence123")

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

    def test_localised_web_assets_are_served(self):
        page = self.client.get("/")
        self.assertIn('id="languageSelect"', page.text)
        self.assertIn('id="guideButton"', page.text)
        self.assertIn('id="guideDialog"', page.text)
        self.assertIn('id="batchProposeButton"', page.text)
        self.assertIn('id="wspAuditButton"', page.text)
        self.assertIn('id="wspSaveButton"', page.text)
        self.assertIn('id="wspSessionButton"', page.text)
        self.assertIn('id="wspSessionSaveButton"', page.text)
        self.assertIn('id="openwspExportButton"', page.text)
        self.assertIn('id="bindingVerifyButton"', page.text)
        self.assertIn('id="firstEditCard"', page.text)
        self.assertIn('id="firstEditButton"', page.text)
        self.assertIn('id="firstEditStarter"', page.text)
        self.assertIn('value="qwen05b"', page.text)
        self.assertIn('data-example="The Eiffel Tower is in the city of -> Rome"', page.text)
        self.assertIn('data-example="The capital of France is -> Paris"', page.text)
        self.assertIn('id="firstEditSave"', page.text)
        self.assertIn('id="firstEditOpen"', page.text)
        self.assertIn('id="firstEditList"', page.text)
        self.assertIn('id="firstEditEvidence"', page.text)
        self.assertIn('id="checkPanel"', page.text)
        self.assertIn('id="checkSentence"', page.text)
        self.assertIn('id="programPanel"', page.text)
        self.assertIn('id="programText"', page.text)
        self.assertIn('id="programRunButton"', page.text)
        self.assertIn('id="fetchOpen"', page.text)
        self.assertIn('data-i18n="onboarding.title"', page.text)
        self.assertIn('src="/static/locales.js"', page.text)
        app = self.client.get("/static/app.js")
        self.assertEqual(app.status_code, 200)
        self.assertIn('state.firstEdit.phase === "fetch"', app.text)
        self.assertIn('state.firstEdit.phase === "verify"', app.text)
        self.assertIn('function nextCheckpointPath', app.text)
        self.assertIn('function manyCheckpointPath', app.text)
        self.assertIn('function renderProgram', app.text)
        self.assertIn('machaon-last-checkpoint', app.text)
        locales = self.client.get("/static/locales.js")
        self.assertEqual(locales.status_code, 200)
        self.assertIn('"zh": {', locales.text)
        self.assertIn('"ja": {', locales.text)
        self.assertIn('"ko": {', locales.text)
        self.assertIn('"firstEdit.title"', locales.text)
        self.assertIn('"firstEdit.readyCheckpoint"', locales.text)
        self.assertIn('"firstEdit.save"', locales.text)
        self.assertIn('"firstEdit.openSaved"', locales.text)
        self.assertIn('"firstEdit.again"', locales.text)
        self.assertIn('"firstEdit.list"', locales.text)
        self.assertIn('"firstEdit.starterQwen"', locales.text)
        self.assertIn('"firstEdit.examples"', locales.text)
        self.assertIn('"firstEdit.evidenceLayer"', locales.text)
        self.assertIn('"firstEdit.verifying"', locales.text)
        self.assertIn('"firstEdit.verified"', locales.text)
        self.assertIn('"firstEdit.verifyFailed"', locales.text)
        self.assertIn('"teachMany.stepLayer"', locales.text)
        self.assertIn('"check.summary"', locales.text)
        self.assertIn('"program.summary"', locales.text)
        self.assertIn('"program.run"', locales.text)

    def test_force_close_rolls_back_an_edited_session(self):
        opened = self.client.post(
            "/api/session/open",
            headers=self.headers,
            json={"path": str(self.root), "dtype": "auto"},
        )
        _wait_for_job(self.client, opened.json()["job"], self.headers)
        applied = self.client.post("/api/edit/apply", headers=self.headers, json={})
        _wait_for_job(self.client, applied.json()["job"], self.headers)
        closed = self.client.post(
            "/api/session/close",
            headers=self.headers,
            json={"force": True},
        )
        result = _wait_for_job(
            self.client, closed.json()["job"], self.headers
        )
        self.assertEqual(result["status"], "done")
        self.assertFalse(result["result"]["ready"])
        state = self.client.get("/api/session/state", headers=self.headers).json()
        self.assertFalse(state["ready"])

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

        fake_adapters = ModuleType("machaon.adapters")
        fake_adapters.Checkpoint = FakeCheckpoint
        with mock.patch.dict(sys.modules, {"machaon.adapters": fake_adapters}):
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

    def test_save_refuses_an_existing_output(self):
        opened = self.client.post(
            "/api/session/open",
            headers=self.headers,
            json={"path": str(self.root), "dtype": "auto"},
        )
        _wait_for_job(self.client, opened.json()["job"], self.headers)
        applied = self.client.post("/api/edit/apply", headers=self.headers, json={})
        _wait_for_job(self.client, applied.json()["job"], self.headers)
        out = self.root / "already-there"
        out.mkdir()
        response = self.client.post(
            "/api/session/save",
            headers=self.headers,
            json={"out_dir": str(out)},
        )
        result = _wait_for_job(self.client, response.json()["job"], self.headers)
        self.assertEqual(result["status"], "error")
        self.assertIn("already exists", result["error"])

    def test_failed_save_removes_the_partial_output(self):
        opened = self.client.post(
            "/api/session/open",
            headers=self.headers,
            json={"path": str(self.root), "dtype": "auto"},
        )
        _wait_for_job(self.client, opened.json()["job"], self.headers)
        applied = self.client.post("/api/edit/apply", headers=self.headers, json={})
        _wait_for_job(self.client, applied.json()["job"], self.headers)

        class FailingCheckpoint:
            def __init__(self, path):
                self.path = path

            def write_tensor(self, name, values, out_dir):
                Path(out_dir).mkdir(parents=True)
                (Path(out_dir) / "partial.safetensors").write_text("x")
                raise OSError("disk full")

        fake_adapters = ModuleType("machaon.adapters")
        fake_adapters.Checkpoint = FailingCheckpoint
        out = self.root / "failed-save"
        with mock.patch.dict(sys.modules, {"machaon.adapters": fake_adapters}):
            response = self.client.post(
                "/api/session/save",
                headers=self.headers,
                json={"out_dir": str(out)},
            )
            result = _wait_for_job(
                self.client, response.json()["job"], self.headers
            )
        self.assertEqual(result["status"], "error")
        self.assertIn("disk full", result["error"])
        self.assertFalse(out.exists())

    def test_export_openwsp_binds_evidence(self):
        opened = self.client.post(
            "/api/session/open",
            headers=self.headers,
            json={"path": str(self.root), "dtype": "auto"},
        )
        _wait_for_job(self.client, opened.json()["job"], self.headers)
        applied = self.client.post("/api/edit/apply", headers=self.headers, json={})
        _wait_for_job(self.client, applied.json()["job"], self.headers)
        response = self.client.post(
            "/api/session/export_openwsp",
            headers=self.headers,
            json={
                "out_dir": str(self.root / "edit"),
                "evidence_sha256": "evidence123",
            },
        )
        result = _wait_for_job(
            self.client, response.json()["job"], self.headers
        )
        self.assertEqual(result["status"], "done")
        manifest = result["result"]["manifest"]
        self.assertEqual(manifest["edit_sha256"], "edit123")
        self.assertEqual(manifest["machaon"]["evidence_sha256"], "evidence123")


if __name__ == "__main__":
    unittest.main()
