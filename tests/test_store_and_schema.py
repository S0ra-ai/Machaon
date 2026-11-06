import os
import tempfile
import unittest

import numpy as np

from machaon.schema import EditSpec
from machaon.store import DeltaStore, StoreCorrupted
from machaon.telesphorus import diff, load, save
from machaon.transaction import Workspace


class TestStore(unittest.TestCase):
    def test_roundtrip_and_dedupe(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = DeltaStore(tmp)
            delta = np.arange(12, dtype=np.float64).reshape(3, 4)
            sha = store.put(delta, {"op": "rank1", "intent": "unit test"})
            self.assertIn(sha, store)
            np.testing.assert_array_equal(store.get(sha), delta)
            manifest = store.manifest(sha)
            self.assertEqual(manifest["shape"], [3, 4])
            self.assertEqual(manifest["intent"], "unit test")
            self.assertEqual(store.put(delta, {"op": "rank1"}), sha)
            self.assertEqual(len(store.list()), 1)

    def test_get_verifies_the_content_address(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = DeltaStore(tmp)
            sha = store.put(np.eye(3), {"op": "rank1"})
            path = store._path(sha)
            with open(path, "r+b") as handle:
                handle.seek(-1, os.SEEK_END)
                handle.write(b"\x00")
            with self.assertRaises(StoreCorrupted):
                store.get(sha)

    def test_invalid_address_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = DeltaStore(tmp)
            with self.assertRaises(ValueError):
                store.get("../../etc/passwd")
            self.assertNotIn("../", store)
            self.assertNotIn("not-a-sha", store)

    def test_non_finite_delta_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = DeltaStore(tmp)
            bad = np.zeros((2, 2))
            bad[0, 0] = np.nan
            with self.assertRaises(ValueError):
                store.put(bad, {"op": "rank1"})

    def test_missing_manifest_is_repaired_on_put(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = DeltaStore(tmp)
            delta = np.eye(2)
            sha = store.put(delta, {"op": "rank1"})
            os.unlink(store._mpath(sha))
            self.assertEqual(store.put(delta, {"op": "rank1"}), sha)
            self.assertEqual(store.manifest(sha)["op"], "rank1")


class TestSchema(unittest.TestCase):
    def test_valid_spec(self):
        spec = EditSpec.from_dict({
            "op": "rank1_constrained",
            "target": {"module": "model.layers.14.mlp.down_proj", "layer": 14},
            "intent": "make France's capital Lyon",
            "constraints": {"max_rel_norm": 0.02},
        }).validate()
        self.assertEqual(spec.target.layer, 14)
        self.assertEqual(spec.constraints.max_rel_norm, 0.02)

    def test_unknown_op_rejected(self):
        spec = EditSpec.from_dict({
            "op": "rewrite_everything", "target": {"module": "x"}, "intent": "y"})
        with self.assertRaises(ValueError):
            spec.validate()

    def test_missing_intent_rejected(self):
        spec = EditSpec.from_dict({"op": "rank1", "target": {"module": "x"}})
        with self.assertRaises(ValueError):
            spec.validate()

    def test_budget_out_of_range_rejected(self):
        spec = EditSpec.from_dict({
            "op": "rank1", "target": {"module": "x"}, "intent": "y",
            "constraints": {"max_rel_norm": 9.0}})
        with self.assertRaises(ValueError):
            spec.validate()


class TestSnapshot(unittest.TestCase):
    def test_save_load_diff(self):
        ws = Workspace()
        ws.register("a", np.eye(4))
        ws.register("b", np.ones((3, 2)))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "snap.npz")
            save(ws, path)
            restored = load(path)
            self.assertEqual(restored.names, ws.names)
            np.testing.assert_array_equal(restored.weight("a"), ws.weight("a"))

            txn = ws.begin()
            txn.apply("a", np.ones((4, 4)))
            txn.commit()
            report = diff(restored, ws)
            self.assertTrue(report["a"]["changed"])
            self.assertFalse(report["b"]["changed"])


if __name__ == "__main__":
    unittest.main()
