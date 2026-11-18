"""The plain-language front end stays explicit and reviewable."""
import unittest

from machaon.intent import IntentError, parse_intent


class TestParseIntent(unittest.TestCase):
    def test_arrow_form_is_exact(self):
        intent = parse_intent("The Eiffel Tower is in the city of -> Rome")
        self.assertEqual(intent.prompt, "The Eiffel Tower is in the city of")
        self.assertEqual(intent.answer, "Rome")
        self.assertEqual(intent.source, "arrow")
        self.assertEqual(intent.confidence, "high")

    def test_chinese_arrow_with_prefix(self):
        intent = parse_intent("教会模型：北京的首都是 → 北京")
        self.assertEqual(intent.prompt, "北京的首都是")
        self.assertEqual(intent.answer, "北京")

    def test_unicode_arrow_is_supported(self):
        intent = parse_intent("水的化学式是 ⇒ H2O")
        self.assertEqual(intent.prompt, "水的化学式是")
        self.assertEqual(intent.answer, "H2O")

    def test_labelled_block_preserves_one_leading_space(self):
        intent = parse_intent(
            "prompt: The Eiffel Tower is in the city of\nanswer:  Rome"
        )
        self.assertEqual(intent.prompt, "The Eiffel Tower is in the city of")
        self.assertEqual(intent.answer, " Rome")
        self.assertEqual(intent.source, "labels")

    def test_labelled_block_keeps_quoted_value_exactly(self):
        intent = parse_intent('prompt: " a  b "\nanswer: "  Rome "')
        self.assertEqual(intent.prompt, " a  b ")
        self.assertEqual(intent.answer, "  Rome ")

    def test_labelled_block_accepts_chinese_labels(self):
        intent = parse_intent("提示词：北京的首都是\n回答: 北京")
        self.assertEqual(intent.prompt, "北京的首都是")
        self.assertEqual(intent.answer, "北京")

    def test_english_phrase_form(self):
        intent = parse_intent(
            'when asked "The Eiffel Tower is in the city of", answer " Rome"'
        )
        self.assertEqual(intent.prompt, "The Eiffel Tower is in the city of")
        self.assertEqual(intent.answer, " Rome")
        self.assertEqual(intent.source, "phrase")

    def test_english_reverse_phrase_form(self):
        intent = parse_intent(
            'teach the model to answer " Rome" when asked '
            '"The Eiffel Tower is in the city of"'
        )
        self.assertEqual(intent.prompt, "The Eiffel Tower is in the city of")
        self.assertEqual(intent.answer, " Rome")

    def test_chinese_phrase_form(self):
        intent = parse_intent("当问到「北京的首都是」时，回答「北京」")
        self.assertEqual(intent.prompt, "北京的首都是")
        self.assertEqual(intent.answer, "北京")

    def test_chinese_teach_time_form(self):
        intent = parse_intent("让模型在「水的化学式是」时回答「H2O」")
        self.assertEqual(intent.prompt, "水的化学式是")
        self.assertEqual(intent.answer, "H2O")

    def test_chinese_teach_comma_form(self):
        intent = parse_intent("教模型「水的化学式是」，回答「H2O」")
        self.assertEqual(intent.prompt, "水的化学式是")
        self.assertEqual(intent.answer, "H2O")

    def test_sentence_form_splits_at_copula_and_is_medium_confidence(self):
        intent = parse_intent("teach the model that the Eiffel Tower is in Paris")
        self.assertEqual(intent.prompt, "the Eiffel Tower is")
        self.assertEqual(intent.answer, " in Paris")
        self.assertEqual(intent.confidence, "medium")
        self.assertTrue(intent.notes)

    def test_chinese_sentence_form_splits_at_shi(self):
        intent = parse_intent("教会模型北京是中国的首都")
        self.assertEqual(intent.prompt, "北京是")
        self.assertEqual(intent.answer, "中国的首都")

    def test_terminal_punctuation_is_dropped_outside_quotes(self):
        intent = parse_intent('when asked "The capital is", answer "Paris".')
        self.assertEqual(intent.answer, "Paris")

    def test_terminal_punctuation_inside_quotes_is_kept(self):
        intent = parse_intent('when asked "Finish this", answer "!"')
        self.assertEqual(intent.answer, "!")

    def test_empty_request_fails_with_examples(self):
        with self.assertRaises(IntentError) as caught:
            parse_intent("   ")
        self.assertIn("Try one of", str(caught.exception))

    def test_an_oversized_request_is_refused(self):
        with self.assertRaises(IntentError) as caught:
            parse_intent("x" * 8001)
        self.assertIn("too long", str(caught.exception))

    def test_unstructured_text_fails_with_examples(self):
        with self.assertRaises(IntentError) as caught:
            parse_intent("please make the model better")
        self.assertIn("could not read", str(caught.exception))
        self.assertTrue(caught.exception.examples)

    def test_forget_is_refused_with_a_specific_message(self):
        with self.assertRaises(IntentError) as caught:
            parse_intent("forget that Paris is in France")
        self.assertIn("teach requests only", str(caught.exception))

    def test_arrow_with_missing_side_fails(self):
        with self.assertRaises(IntentError):
            parse_intent("The capital is ->")
        with self.assertRaises(IntentError):
            parse_intent("-> Paris")

    def test_to_dict_and_command_are_stable(self):
        intent = parse_intent("teach the model that Paris is in France")
        payload = intent.to_dict()
        self.assertEqual(payload["kind"], "teach")
        self.assertEqual(payload["confidence"], "medium")
        command = intent.to_command("/tmp/model dir")
        self.assertIn("--prompt", command)
        self.assertIn("'/tmp/model dir'", command)


if __name__ == "__main__":
    unittest.main()
