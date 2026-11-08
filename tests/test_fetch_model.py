"""Endpoint fallback for model downloads stays deterministic and testable."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from fetch_model import download_with_fallback  # noqa: E402


class TestDownloadFallback(unittest.TestCase):
    def test_tries_next_endpoint_after_a_failure(self):
        calls = []

        def fake_download(repo, name, target, endpoint):
            calls.append(endpoint)
            if endpoint == "first":
                raise IOError("403")
            target.write_bytes(b"weights")

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "model.safetensors"
            label, endpoint = download_with_fallback(
                "org/model",
                "model.safetensors",
                target,
                [("mirror", "first"), ("origin", "second")],
                downloader=fake_download,
            )
            self.assertEqual(calls, ["first", "second"])
            self.assertEqual(label, "origin")
            self.assertEqual(endpoint, "second")
            self.assertEqual(target.read_bytes(), b"weights")

    def test_invalid_payload_falls_through_to_next_endpoint(self):
        calls = []

        def fake_download(repo, name, target, endpoint):
            calls.append(endpoint)
            target.write_bytes(b"bad" if endpoint == "first" else b"good")

        def validate(path):
            if path.read_bytes() != b"good":
                raise ValueError("bad header")

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "model.safetensors"
            download_with_fallback(
                "org/model",
                "model.safetensors",
                target,
                [("mirror", "first"), ("origin", "second")],
                downloader=fake_download,
                validator=validate,
            )
            self.assertEqual(calls, ["first", "second"])
            self.assertEqual(target.read_bytes(), b"good")

    def test_reports_every_endpoint_when_all_fail(self):
        def failing_download(repo, name, target, endpoint):
            raise IOError("unavailable")

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(IOError) as raised:
                download_with_fallback(
                    "org/model",
                    "model.safetensors",
                    Path(tmp) / "model.safetensors",
                    [("a", "first"), ("b", "second")],
                    downloader=failing_download,
                )
        self.assertIn("a: unavailable", str(raised.exception))
        self.assertIn("b: unavailable", str(raised.exception))

    def test_forced_endpoint_is_a_single_candidate(self):
        calls = []

        def fake_download(repo, name, target, endpoint):
            calls.append(endpoint)
            target.write_bytes(b"x")

        with tempfile.TemporaryDirectory() as tmp:
            download_with_fallback(
                "org/model",
                "config.json",
                Path(tmp) / "config.json",
                [("only", "forced")],
                downloader=fake_download,
            )
        self.assertEqual(calls, ["forced"])


if __name__ == "__main__":
    unittest.main()
