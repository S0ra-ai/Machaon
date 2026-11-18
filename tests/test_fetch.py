"""The starter downloader is deterministic and does not need torch."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from machaon import fetch


class TestRecommendations(unittest.TestCase):
    def test_every_starter_has_a_repo_and_display_name(self):
        for key, item in fetch.RECOMMENDED.items():
            self.assertTrue(key)
            self.assertTrue(item["repo"])
            self.assertTrue(item["name"])
            self.assertTrue(item["size"])

    def test_list_outputs_the_catalogue(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = fetch.main(["--list"])
        self.assertEqual(code, 0)
        self.assertIn("qwen05b", out.getvalue())
        self.assertIn("Qwen/Qwen2.5-0.5B", out.getvalue())

    def test_list_json_is_machine_readable(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = fetch.main(["--list", "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(out.getvalue())
        self.assertIn("qwen05b", payload)
        self.assertEqual(payload["qwen05b"]["repo"], "Qwen/Qwen2.5-0.5B")


class TestFileSelection(unittest.TestCase):
    def test_weights_and_tokenizer_files_are_wanted(self):
        for name in (
            "model.safetensors",
            "model-00001-of-00002.safetensors",
            "config.json",
            "generation_config.json",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "tokenizer.model",
            "sentencepiece.bpe.model",
            "chat_template.jinja",
            "vocab.json",
            "merges.txt",
        ):
            self.assertTrue(fetch._wanted_file(name), name)

    def test_experiment_logs_are_not_wanted(self):
        for name in (
            "README.md",
            "eval_results.json",
            "train_results.json",
            "trainer_state.json",
            "all_results.json",
            "configuration.json",
            "pytorch_model.bin",
        ):
            self.assertFalse(fetch._wanted_file(name), name)


class TestFetchModel(unittest.TestCase):
    def _fake_download(self, target_bytes=b"weights"):
        def downloader(repo, name, target, endpoint):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(target_bytes)

        return downloader

    def test_downloads_wanted_files_and_reports_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "model"
            files = [
                "config.json",
                "model.safetensors",
                "tokenizer.json",
                "README.md",
                ".gitattributes",
            ]
            with mock.patch.object(fetch, "list_files", lambda repo, endpoint: files), \
                    mock.patch.object(fetch, "download", self._fake_download()), \
                    mock.patch.object(
                        fetch, "check_safetensors",
                        lambda path: {"tensors": 2, "parameters": 100},
                    ):
                report = fetch.fetch_model("org/model", dest)
            self.assertEqual(report["repo"], "org/model")
            self.assertEqual(report["endpoint"], "huggingface")
            self.assertIn("model.safetensors", report["files"])
            self.assertIn("config.json", report["files"])
            self.assertNotIn("README.md", report["files"])
            self.assertEqual(report["parameters"], 100)
            self.assertTrue((dest / "model.safetensors").exists())

    def test_progress_reports_list_plan_file_download_and_done(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "model"
            events = []

            def fake_fallback(repo, name, target, candidates,
                              downloader=None, validator=None, progress=None):
                if progress:
                    progress({
                        "phase": "download",
                        "file": name,
                        "bytes": 1,
                        "total": 2,
                    })
                target.write_bytes(b"x")
                return "huggingface", candidates[0][1]

            with mock.patch.object(
                fetch, "list_files",
                lambda repo, endpoint: ["model.safetensors"],
            ), mock.patch.object(
                fetch, "download_with_fallback", fake_fallback
            ), mock.patch.object(
                fetch, "check_safetensors",
                lambda path: {"tensors": 1, "parameters": 10},
            ):
                fetch.fetch_model(
                    "org/model", dest, progress=events.append
                )
            phases = [event["phase"] for event in events]
            self.assertEqual(
                phases, ["list", "plan", "file", "download", "done"]
            )
            self.assertEqual(events[-1]["parameters"], 10)

    def test_existing_files_are_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "model"
            dest.mkdir()
            (dest / "model.safetensors").write_bytes(b"cached")
            calls = []

            def downloader(repo, name, target, endpoint):
                calls.append(name)
                target.write_bytes(b"new")

            with mock.patch.object(
                fetch, "list_files",
                lambda repo, endpoint: ["config.json", "model.safetensors"],
            ), mock.patch.object(fetch, "download", downloader), \
                    mock.patch.object(
                        fetch, "check_safetensors",
                        lambda path: {"tensors": 1, "parameters": 10},
                    ):
                report = fetch.fetch_model("org/model", dest)
            self.assertEqual(calls, ["config.json"])
            self.assertIn("model.safetensors", report["cached"])
            self.assertEqual((dest / "model.safetensors").read_bytes(), b"cached")

    def test_require_new_refuses_a_non_empty_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "model"
            dest.mkdir()
            (dest / "old.txt").write_text("x")
            with self.assertRaises(FileExistsError):
                fetch.fetch_model("org/model", dest, require_new=True)

    def test_require_new_allows_an_empty_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "model"
            dest.mkdir()
            with mock.patch.object(
                fetch, "list_files",
                lambda repo, endpoint: ["model.safetensors"],
            ), mock.patch.object(fetch, "download", self._fake_download()), \
                    mock.patch.object(
                        fetch, "check_safetensors",
                        lambda path: {"tensors": 1, "parameters": 10},
                    ):
                report = fetch.fetch_model(
                    "org/model", dest, require_new=True
                )
            self.assertEqual(report["tensors"], 1)

    def test_an_index_with_a_missing_shard_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "model"
            dest.mkdir()
            (dest / "model.safetensors").write_bytes(b"shard")
            (dest / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {"a": "model.safetensors",
                               "b": "model-00002.safetensors"}
            }))
            with mock.patch.object(
                fetch, "list_files",
                lambda repo, endpoint: ["model.safetensors"],
            ), mock.patch.object(
                fetch, "check_safetensors",
                lambda path: {"tensors": 1, "parameters": 10},
            ):
                with self.assertRaises(ValueError) as raised:
                    fetch.fetch_model("org/model", dest)
            self.assertIn("missing shards", str(raised.exception))

    def test_a_repo_without_safetensors_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(
                fetch, "list_files",
                lambda repo, endpoint: ["config.json", "pytorch_model.bin"],
            ):
                with self.assertRaises(ValueError) as raised:
                    fetch.fetch_model("org/model", Path(tmp) / "model")
            self.assertIn("no .safetensors", str(raised.exception))

    def test_unknown_endpoint_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                fetch.fetch_model(
                    "org/model", Path(tmp) / "model", endpoint="nowhere"
                )

    def test_listing_errors_report_every_endpoint(self):
        def fail(repo, endpoint):
            raise OSError("offline")

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(fetch, "list_files", fail):
            with self.assertRaises(IOError) as raised:
                fetch.fetch_model("org/model", Path(tmp) / "model")
        self.assertIn("huggingface", str(raised.exception))
        self.assertIn("modelscope", str(raised.exception))


class TestCli(unittest.TestCase):
    def test_fetch_requires_a_source(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = fetch.main([])
        self.assertEqual(code, 2)
        self.assertIn("--repo", err.getvalue())

    def test_starter_uses_the_catalogue_and_default_destination(self):
        calls = []

        def fake(repo, dest, endpoint=None, require_new=False):
            calls.append((repo, dest, endpoint, require_new))
            return {
                "repo": repo,
                "dest": str(dest),
                "endpoint": "huggingface",
                "shards": 1,
                "tensors": 1,
                "parameters": 10,
            }

        with mock.patch.object(fetch, "fetch_model", fake):
            code = fetch.main(["--starter", "qwen05b"])
        self.assertEqual(code, 0)
        self.assertEqual(calls[0][0], "Qwen/Qwen2.5-0.5B")
        self.assertIn("models", str(calls[0][1]))


if __name__ == "__main__":
    unittest.main()
