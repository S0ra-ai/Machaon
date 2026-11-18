"""Multi-fact teaching keeps a chain of safe checkpoints."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from machaon.recipes import TeachPolicy
from machaon.teach_many import load_facts, teach_many


class FakeRuntime:
    def __init__(self):
        self.shift = 0.0

    def next_logits(self, prompts):
        rows = []
        for _ in prompts:
            row = np.full(8, -10.0)
            row[1 if self.shift >= 1.0 else 0] = 10.0
            rows.append(row)
        return np.stack(rows)


class FakeSession:
    def __init__(self):
        self.runtime = FakeRuntime()


class FakeTeachResult:
    def __init__(self, status, delta=0.0):
        self.status = status
        self.relative_delta = delta

    def to_dict(self):
        return {"status": self.status, "relative_delta": self.relative_delta}


class ScriptedTeach:
    def __init__(self, statuses, deltas=None, shifts=None):
        self.statuses = list(statuses)
        self.deltas = list(deltas or [0.0] * len(statuses))
        self.shifts = list(shifts or [0.0] * len(statuses))
        self.calls = []

    def __call__(self, session, prompt, answer, **kwargs):
        index = len(self.calls)
        self.calls.append((prompt, answer, kwargs))
        session.runtime.shift = self.shifts[index]
        return FakeTeachResult(self.statuses[index], self.deltas[index])


def fake_save(session, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}")
    return {"output": str(path)}


def make_factory(calls):
    def factory(path, budget):
        calls.append((str(path), budget.max_total))
        return FakeSession()

    return factory


class TestLoadFacts(unittest.TestCase):
    def test_text_lines_use_the_intent_grammar(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "facts.txt"
            path.write_text(
                "# a comment\n"
                "The Eiffel Tower is in the city of -> Rome\n"
                "北京的首都是 -> 北京\n"
            )
            facts = load_facts(path)
        self.assertEqual(facts, [
            ("The Eiffel Tower is in the city of", "Rome"),
            ("北京的首都是", "北京"),
        ])

    def test_json_strings_objects_and_say(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "facts.json"
            path.write_text(json.dumps({
                "facts": [
                    "The capital is -> Paris",
                    {"prompt": "The city is", "answer": " Rome"},
                    {"say": "The river is -> the Seine"},
                ]
            }))
            facts = load_facts(path)
        self.assertEqual(facts, [
            ("The capital is", "Paris"),
            ("The city is", " Rome"),
            ("The river is", "the Seine"),
        ])

    def test_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "facts.jsonl"
            path.write_text(
                '{"say": "The capital is -> Paris"}\n'
                '{"prompt": "The city is", "answer": " Rome"}\n'
            )
            facts = load_facts(path)
        self.assertEqual(len(facts), 2)
        self.assertEqual(facts[0], ("The capital is", "Paris"))

    def test_invalid_json_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "facts.json"
            path.write_text("{not json")
            with self.assertRaises(ValueError):
                load_facts(path)

    def test_empty_file_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "facts.txt"
            path.write_text("\n# only a comment\n")
            with self.assertRaises(ValueError):
                load_facts(path)


class TestTeachMany(unittest.TestCase):
    def _run(self, statuses, deltas=None, shifts=None, **kwargs):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        out = Path(self.tmp.name) / "batch"
        calls = []
        teach = ScriptedTeach(statuses, deltas, shifts)
        facts = [("p1", " a"), ("p2", " b"), ("p3", " c")][: len(statuses)]
        result = teach_many(
            make_factory(calls),
            "base",
            facts,
            out,
            teach_one=teach,
            save=fake_save,
            **kwargs,
        )
        return result, out, calls, teach

    def test_all_facts_applied_and_final_promoted(self):
        result, out, calls, teach = self._run(
            ["applied", "applied", "applied"],
            [0.01, 0.02, 0.03],
        )
        self.assertEqual(result.status, "applied")
        self.assertTrue(result.accepted)
        self.assertEqual(result.cumulative_relative_delta, 0.06)
        self.assertEqual(result.final, str(out / "final"))
        self.assertTrue((out / "final" / "config.json").exists())
        self.assertFalse((out / "step-001").exists())
        self.assertFalse((out / "step-002").exists())
        self.assertTrue((out / "report.json").exists())
        for actual, expected in zip([call[1] for call in calls], [0.10, 0.09, 0.07]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(len(teach.calls), 3)
        self.assertEqual(result.steps[-1].checkpoint, str(out / "final"))
        report = json.loads((out / "report.json").read_text())
        self.assertEqual(report["final"], str(out / "final"))

    def test_rejected_fact_stops_at_the_last_safe_checkpoint(self):
        result, out, _calls, teach = self._run(
            ["applied", "rejected", "applied"],
            [0.01, 0.0, 0.01],
        )
        self.assertEqual(result.status, "stopped")
        self.assertEqual(result.stopped_at, 2)
        self.assertEqual(len(result.steps), 2)
        self.assertTrue((out / "final" / "config.json").exists())
        self.assertIsNone(result.steps[1].checkpoint)
        self.assertEqual(len(teach.calls), 2)

    def test_global_drift_removes_the_failed_step(self):
        result, out, _calls, _teach = self._run(
            ["applied", "applied"],
            [0.01, 0.01],
            shifts=[0.0, 1.0],
        )
        self.assertEqual(result.status, "global_reject")
        self.assertEqual(result.stopped_at, 2)
        self.assertAlmostEqual(result.cumulative_relative_delta, 0.01)
        self.assertEqual(result.final, str(out / "final"))
        self.assertTrue((out / "final" / "config.json").exists())
        self.assertFalse(result.steps[1].kept)
        self.assertFalse(result.steps[1].global_passed)

    def test_budget_exhaustion_stops_before_the_next_fact(self):
        result, out, calls, teach = self._run(
            ["applied", "applied"],
            [0.02, 0.02],
            budget=0.02,
        )
        self.assertEqual(result.status, "budget_exhausted")
        self.assertEqual(result.stopped_at, 2)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(teach.calls), 1)
        self.assertTrue((out / "final").exists())

    def test_already_true_does_not_save(self):
        result, out, _calls, _teach = self._run(
            ["already_true", "applied"],
            [0.0, 0.01],
        )
        self.assertEqual(result.status, "applied")
        self.assertIsNone(result.steps[0].checkpoint)
        self.assertEqual(result.steps[1].checkpoint, str(out / "final"))

    def test_continue_on_reject_keeps_going(self):
        result, out, _calls, teach = self._run(
            ["applied", "rejected", "applied"],
            [0.01, 0.0, 0.01],
            continue_on_reject=True,
        )
        self.assertEqual(result.status, "applied")
        self.assertEqual(len(teach.calls), 3)
        self.assertEqual(result.final, str(out / "final"))
        self.assertIsNone(result.steps[1].checkpoint)

    def test_keep_steps_keeps_the_whole_chain(self):
        result, out, _calls, _teach = self._run(
            ["applied", "applied"],
            [0.01, 0.01],
            keep_steps=True,
        )
        self.assertEqual(result.final, str(out / "step-002"))
        self.assertTrue((out / "step-001" / "config.json").exists())
        self.assertTrue((out / "step-002" / "config.json").exists())
        self.assertFalse((out / "final").exists())

    def test_no_fact_rejected_has_no_final(self):
        result, out, _calls, _teach = self._run(["rejected"], [0.0])
        self.assertEqual(result.status, "stopped")
        self.assertIsNone(result.final)
        self.assertIn("No facts were applied", result.summary)
        self.assertFalse((out / "final").exists())

    def test_all_already_true_is_no_edits(self):
        result, _out, _calls, _teach = self._run(
            ["already_true", "already_true"], [0.0, 0.0]
        )
        self.assertEqual(result.status, "no_edits")
        self.assertIn("already true", result.reason)

    def test_teach_error_stops_the_chain(self):
        def explode(session, prompt, answer, **kwargs):
            raise RuntimeError("solver failed")

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        out = Path(self.tmp.name) / "batch"
        result = teach_many(
            make_factory([]),
            "base",
            [("p1", " a")],
            out,
            teach_one=explode,
            save=fake_save,
        )
        self.assertEqual(result.status, "error")
        self.assertEqual(result.steps[0].status, "error")
        self.assertIsNone(result.final)

    def test_custom_policy_is_forwarded(self):
        policy = TeachPolicy(max_rel_norm=0.01, min_panel_agreement=1.0)
        _result, _out, _calls, teach = self._run(
            ["applied"], [0.005], policy=policy
        )
        forwarded = teach.calls[0][2]["policy"]
        self.assertEqual(forwarded.max_rel_norm, 0.01)
        self.assertEqual(forwarded.min_panel_agreement, 1.0)


if __name__ == "__main__":
    unittest.main()
