import os
import tempfile
import unittest

import numpy as np

from machaon.schema import EditSpec
from machaon.store import DeltaStore
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
