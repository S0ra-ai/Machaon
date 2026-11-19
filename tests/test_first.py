"""The zero-to-one command composes fetch, teach and save."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from machaon.first import first_edit
from machaon.intent import IntentError


class FakeTeachResult:
    def __init__(self, status, delta=0.0):
        self.status = status
        self.relative_delta = delta

    def to_dict(self):
        return {"status": self.status, "relative_delta": self.relative_delta}


def fake_teach(status, delta=0.0, calls=None):
    def teach(session, prompt, answer, **kwargs):
        if calls is not None:
            calls.append((prompt, answer, kwargs))
        return FakeTeachResult(status, delta)

    return teach


class TestFirstEdit(unittest.TestCase):
    def test_missing_runtime_fails_before_downloading(self):
        with tempfile.TemporaryDirectory() as tmp:
            fetched = []

            def fetch(*args, **kwargs):
                fetched.append(args)

            with mock.patch("importlib.util.find_spec", return_value=None):
                with self.assertRaisesRegex(ImportError, "machaon\\[hf\\]"):
                    first_edit(
                        "The capital is -> Paris",
                        Path(tmp) / "model",
                        Path(tmp) / "out",
                        fetch_fn=fetch,
                    )
            self.assertEqual(fetched, [])

    def test_missing_model_fetches_starter_and_saves(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            out = root / "edited"
            fetched = []
            saved = []

            def fetch(repo, dest, **kwargs):
                fetched.append((repo, str(dest), kwargs))
                Path(dest).mkdir(parents=True)
                return {"repo": repo, "dest": str(dest)}

            def save(session, path):
                saved.append(str(path))
                Path(path).mkdir(parents=True)
                return {"output": str(path)}

            result = first_edit(
                "The capital is -> Paris",
                model,
                out,
                fetch_fn=fetch,
                teach_one=fake_teach("applied", 0.01),
                save_fn=save,
                session_factory=lambda path, budget: object(),
            )
            self.assertTrue(result.model_fetched)
            self.assertEqual(fetched[0][0], "HuggingFaceTB/SmolLM2-135M-Instruct")
            self.assertEqual(saved, [str(out)])
            self.assertEqual(result.final, str(out))
            self.assertTrue(result.accepted)
            self.assertIn("Downloaded", result.summary)

    def test_existing_model_skips_fetch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            model.mkdir()
            calls = []

            result = first_edit(
                "The capital is -> Paris",
                model,
                None,
                fetch_fn=lambda *a, **k: self.fail("fetch must not run"),
                teach_one=fake_teach("applied", calls=calls),
                save_fn=lambda *a, **k: self.fail("save must not run"),
                session_factory=lambda path, budget: object(),
                no_save=True,
            )
            self.assertFalse(result.model_fetched)
            self.assertEqual(result.final, None)
            self.assertIn("in memory", result.summary)

    def test_already_true_does_not_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = first_edit(
                "The capital is -> Paris",
                Path(tmp),
                Path(tmp) / "out",
                teach_one=fake_teach("already_true"),
                save_fn=lambda *a, **k: self.fail("save must not run"),
                session_factory=lambda path, budget: object(),
            )
        self.assertEqual(result.status, "already_true")
        self.assertIsNone(result.final)
        self.assertTrue(result.accepted)
        self.assertIn("already knew", result.summary)

    def test_rejected_edit_does_not_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = first_edit(
                "The capital is -> Paris",
                Path(tmp),
                Path(tmp) / "out",
                teach_one=fake_teach("rejected"),
                save_fn=lambda *a, **k: self.fail("save must not run"),
                session_factory=lambda path, budget: object(),
            )
        self.assertEqual(result.status, "rejected")
        self.assertFalse(result.accepted)
        self.assertIsNone(result.final)
        self.assertIn("rejected", result.summary)

    def test_no_save_keeps_the_edit_in_memory_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = first_edit(
                "The capital is -> Paris",
                Path(tmp),
                Path(tmp) / "out",
                teach_one=fake_teach("applied", 0.01),
                save_fn=lambda *a, **k: self.fail("save must not run"),
                session_factory=lambda path, budget: object(),
                no_save=True,
            )
        self.assertEqual(result.status, "applied")
        self.assertIsNone(result.final)
        self.assertIn("in memory", result.summary)

    def test_unknown_starter_refuses_a_missing_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                first_edit(
                    "The capital is -> Paris",
                    Path(tmp) / "missing",
                    None,
                    starter="nope",
                    fetch_fn=lambda *a, **k: None,
                    teach_one=fake_teach("applied"),
                    session_factory=lambda path, budget: object(),
                )

    def test_invalid_sentence_is_refused_before_any_io(self):
        with self.assertRaises(IntentError):
            first_edit(
                "please make it better",
                "not-a-model",
                None,
                fetch_fn=lambda *a, **k: self.fail("fetch must not run"),
                teach_one=fake_teach("applied"),
                session_factory=lambda path, budget: object(),
            )

    def test_budget_and_policy_are_forwarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            budgets = []
            calls = []

            def factory(path, budget):
                budgets.append(budget.max_total)
                return object()

            first_edit(
                "The capital is -> Paris",
                Path(tmp),
                None,
                budget=0.05,
                teach_one=fake_teach("applied", calls=calls),
                session_factory=factory,
                no_save=True,
            )
        self.assertEqual(budgets, [0.05])
        self.assertEqual(calls[0][0], "The capital is")
        self.assertEqual(calls[0][1], "Paris")
        self.assertEqual(calls[0][2]["policy"].max_rel_norm, 0.02)


if __name__ == "__main__":
    unittest.main()
