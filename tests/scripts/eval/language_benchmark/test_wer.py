# Path: tests/scripts/eval/language_benchmark/test_wer.py

import pytest

from scripts.eval.language_benchmark.wer import align, digit_tokens, normalize, utterance_error_rates


class TestNormalize:
    def test_cyrillic_reference_compares_with_latin_transcript(self):
        assert normalize("У том погледу, Шенгенска зона!") == normalize("u tom pogledu šengenska zona")

    def test_punctuation_and_case_are_ignored_but_letters_kept(self):
        assert normalize("Dr. Petrović—rekao je: „Šećer”.") == ["dr", "petrović", "rekao", "je", "šećer"]

    def test_inner_apostrophes_join_the_word(self):
        assert normalize("Don't stop") == ["dont", "stop"]


class TestAlign:
    def test_identical_text_has_no_errors(self):
        result = align(["a", "b", "c"], ["a", "b", "c"])

        assert (result.wer, result.errors, result.reference_correct) == (0.0, 0, [True, True, True])

    def test_counts_substitution_deletion_and_insertion(self):
        result = align(["the", "cat", "sat", "down"], ["the", "bat", "sat", "down", "now"])

        assert (result.substitutions, result.deletions, result.insertions) == (1, 0, 1)
        assert result.wer == pytest.approx(2 / 4)
        assert result.edits == [("cat", "bat"), (None, "now")]

    def test_deletions_are_attributed_to_the_missing_reference_words(self):
        result = align(["a", "b", "c", "d"], ["a", "d"])

        assert result.deletions == 2
        assert result.reference_correct == [True, False, False, True]

    @pytest.mark.parametrize("ref, hyp, wer", [([], [], 0.0), ([], ["x"], 0.0), (["x", "y"], [], 1.0)])
    def test_empty_sides(self, ref, hyp, wer):
        assert align(ref, hyp).wer == wer


class TestUtteranceRates:
    def test_rates_follow_the_utterance_boundaries(self):
        alignment = align(["a", "b", "c", "d", "e"], ["a", "b", "x", "y", "e"])

        assert utterance_error_rates(alignment, [2, 3]) == [0.0, pytest.approx(2 / 3)]


def test_digit_tokens():
    assert digit_tokens(["20", "dvadeset", "7-2", "v3"]) == 3
