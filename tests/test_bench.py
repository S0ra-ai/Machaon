import tempfile
import unittest
from pathlib import Path

import numpy as np

from machaon.bench import (
    Provenance,
    file_sha256,
    load_mmlu,
    load_wikitext,
    load_zsre,
    mmlu_accuracy,
    read_provenance,
    write_provenance,
)

FIXTURES = Path(__file__).parent / "fixtures" / "bench"


class TestZsre(unittest.TestCase):
    def test_loads_question_answer_and_rephrases(self):
        cases = load_zsre(FIXTURES / "zsre.jsonl")
        self.assertEqual(len(cases), 3)
        first = cases[0]
        self.assertTrue(first.question)
        self.assertTrue(first.answer)
        self.assertGreater(len(first.rephrases), 0)
        self.assertEqual(first.prompts[0], first.question)

    def test_limit(self):
        self.assertEqual(len(load_zsre(FIXTURES / "zsre.jsonl", limit=2)), 2)


class TestWikitext(unittest.TestCase):
    def test_filters_short_documents(self):
        docs = load_wikitext(FIXTURES / "wikitext.parquet")
        self.assertEqual(len(docs), 2)
        self.assertTrue(all(len(d) >= 400 for d in docs))

    def test_limit(self):
        self.assertEqual(len(load_wikitext(FIXTURES / "wikitext.parquet", limit=1)), 1)


class TestMmlu(unittest.TestCase):
    def test_loads_items(self):
        items = load_mmlu(FIXTURES / "mmlu.parquet")
        self.assertEqual(len(items), 3)
        self.assertEqual(items[0].choices[1], "4")
        self.assertEqual(items[0].answer, 1)

    def test_subject_filter(self):
        self.assertEqual(len(load_mmlu(FIXTURES / "mmlu.parquet", subjects=["math"])), 1)

    def test_accuracy_with_a_stub_runtime(self):
        class Perfect:
            def score_continuations(self, pairs):
                # highest score for the letter that matches the known answer
                rule = {"What is 2+2?": " B", "Capital of France?": " A",
                        "Largest planet?": " B"}
                return np.array([1.0 if continuation == rule[p.split("\n")[0]] else 0.0
                                 for p, continuation in pairs])

        items = load_mmlu(FIXTURES / "mmlu.parquet")
        result = mmlu_accuracy(Perfect(), items)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["n"], 3)

    def test_prompt_format(self):
        from machaon.bench.mmlu import format_prompt
        text = format_prompt(load_mmlu(FIXTURES / "mmlu.parquet")[1])
        self.assertIn("A. Paris", text)
        self.assertTrue(text.endswith("Answer:"))


class TestProvenance(unittest.TestCase):
    def test_roundtrip_and_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.jsonl"
            path.write_text("a\nb\n")
            record = Provenance(
                name="toy", source="file://toy", revision="1", license="MIT",
                sha256=file_sha256(path), n_items=2, retrieved="2026-11-04",
            )
            write_provenance(path, record)
            self.assertEqual(read_provenance(path), record)

            from machaon.bench.provenance import verify
            self.assertEqual(verify(path).n_items, 2)

            path.write_text("a\nb\nc\n")
            with self.assertRaises(ValueError):
                verify(path)


if __name__ == "__main__":
    unittest.main()
