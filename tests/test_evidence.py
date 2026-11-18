"""Content-addressed WSP evidence bundles."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from machaon.binding import verify_binding
from machaon.evidence import (
    EvidenceError,
    evidence_sha256,
    is_wsp_evidence,
    read_evidence,
    write_evidence,
)
from machaon.interop import openwsp

REPORT = {
    "accepted": True,
    "label": "repair",
    "reason": "unit test",
    "classification": {"control_gap": 0.7, "headroom_gap": 0.02},
}


class TestEvidence(unittest.TestCase):
    def test_roundtrip_file_and_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            file_path = Path(tmp) / "one.json"
            document = write_evidence(file_path, REPORT, {"base": "m@rev"})
            self.assertEqual(document["format"], "machaon.wsp.evidence")
            self.assertEqual(document["format_version"], 1)
            self.assertEqual(document["evidence_sha256"], evidence_sha256(document))

            loaded = read_evidence(file_path)
            self.assertEqual(loaded["report"], REPORT)
            self.assertEqual(loaded["metadata"]["base"], "m@rev")

            dir_path = Path(tmp) / "bundle"
            write_evidence(dir_path, REPORT)
            self.assertTrue((dir_path / "evidence.json").exists())
            self.assertEqual(read_evidence(dir_path)["report"], REPORT)

    def test_same_content_has_same_address(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = write_evidence(Path(tmp) / "a.json", REPORT)
            b = write_evidence(Path(tmp) / "b.json", REPORT)
            self.assertEqual(a["evidence_sha256"], b["evidence_sha256"])

    def test_tampered_report_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence.json"
            document = write_evidence(path, REPORT)
            document["report"]["label"] = "acquisition"
            path.write_text(json.dumps(document))
            with self.assertRaises(EvidenceError):
                read_evidence(path)

    def test_tampered_metadata_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence.json"
            document = write_evidence(path, REPORT, {"seed": 0})
            document["metadata"]["seed"] = 1
            path.write_text(json.dumps(document))
            with self.assertRaises(EvidenceError):
                read_evidence(path)

    def test_wrong_format_or_version_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence.json"
            document = write_evidence(path, REPORT)
            document["format"] = "other"
            path.write_text(json.dumps(document))
            with self.assertRaises(EvidenceError):
                read_evidence(path)

            document = write_evidence(path, REPORT)
            document["format_version"] = 2
            path.write_text(json.dumps(document))
            with self.assertRaises(EvidenceError):
                read_evidence(path)

    def test_malformed_json_and_non_serialisable_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("{not json")
            with self.assertRaises(EvidenceError):
                read_evidence(path)
            with self.assertRaises(EvidenceError):
                write_evidence(Path(tmp) / "out.json", {"value": object()})

    def test_is_wsp_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence.json"
            write_evidence(path, REPORT)
            self.assertTrue(is_wsp_evidence(path))
            path.write_text("{}")
            self.assertFalse(is_wsp_evidence(path))

    def test_openwsp_evidence_binding(self):
        rng = np.random.default_rng(21)
        base = {"w": rng.normal(size=(4, 3)).astype(np.float32)}
        edited = {"w": base["w"] + 0.01}
        with tempfile.TemporaryDirectory() as tmp:
            evidence = write_evidence(Path(tmp) / "evidence", REPORT)
            edit_dir = Path(tmp) / "edit"
            manifest = openwsp.export_edit(
                base, edited, edit_dir,
                machaon_metadata={
                    "evidence_sha256": evidence["evidence_sha256"],
                    "evidence_format": evidence["format"],
                },
            )
            report = verify_binding(edit_dir, Path(tmp) / "evidence")
            self.assertTrue(report.passed, report.errors)
            self.assertEqual(report.edit_sha256, manifest["edit_sha256"])
            self.assertEqual(report.label, "repair")

    def test_binding_detects_unbound_and_tampered_manifest(self):
        rng = np.random.default_rng(22)
        base = {"w": rng.normal(size=(4, 3)).astype(np.float32)}
        edited = {"w": base["w"] + 0.01}
        with tempfile.TemporaryDirectory() as tmp:
            evidence = write_evidence(Path(tmp) / "evidence", REPORT)
            edit_dir = Path(tmp) / "edit"
            openwsp.export_edit(base, edited, edit_dir)
            report = verify_binding(edit_dir, Path(tmp) / "evidence")
            self.assertFalse(report.passed)
            self.assertIn("no machaon.evidence_sha256", " ".join(report.errors))

            openwsp.export_edit(
                base, edited, edit_dir,
                machaon_metadata={"evidence_sha256": evidence["evidence_sha256"]},
            )
            manifest_path = edit_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["machaon"]["evidence_sha256"] = "tampered"
            manifest_path.write_text(json.dumps(manifest))
            report = verify_binding(edit_dir, Path(tmp) / "evidence")
            self.assertFalse(report.passed)
            self.assertIn("does not match evidence bundle", " ".join(report.errors))


if __name__ == "__main__":
    unittest.main()
