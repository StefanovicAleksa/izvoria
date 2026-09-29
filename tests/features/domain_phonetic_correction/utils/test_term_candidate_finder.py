from typing import List, Optional
from unittest.mock import patch

import pytest

from app.features.domain_phonetic_correction.utils import term_candidate_finder as finder
from app.features.domain_phonetic_correction.utils.term_candidate_finder import find_term_candidates
from app.features.transcription.domain.entities import TranscriptWord

TERMS = ["Bunds", "Mario Draghi", "ECB", "FOMC", "TLTRO", "Forward Guidance", "hawkish", "Deposit Rate",
         "Philip Lane", "Non-farm Payrolls", "acute angina", "Frankfurt", "stagflation"]


def _find(text: str, words: Optional[List[TranscriptWord]] = None, max_word_probability: float = 0.95):
    return find_term_candidates(
        text=text, words=words or [], terms=TERMS, min_similarity=0.7,
        length_ratio_min=0.6, length_ratio_max=1.67, max_word_probability=max_word_probability
    )


def _words(text: str, probabilities: dict) -> List[TranscriptWord]:
    return [TranscriptWord(word=f" {token}", start_sec=float(i), end_sec=i + 0.5,
                           probability=probabilities.get(token.strip(".,"), 0.99))
            for i, token in enumerate(text.split())]


class TestFindTermCandidates:
    @pytest.mark.parametrize("text", [
        # Natural speech Whisper transcribed correctly; the old free-rewrite pass damaged these
        "So again showing that there is still a little bit of weakness in the bund.",
        "You have the fade from the top of the range up at those 73s.",
        "Holding in the 20s is important.",
        "Past three days and including the morning of the Bank of England decision, the market has rallied.",
        "First idea is going to be the very first move.",
        "You want to set your replay for 9th March, starting at 13.50.45 seconds.",
        # Correct terms written with different case, punctuation, hyphens or number
        "Read through the E.C.B. statements from October.",
        "They left the deposit rates unchanged.",
        "Philip Lane spoke about it.",
        "The non farm payrolls number came in strong.",
        "We're looking at the ECB meeting.",
    ])
    def test_no_candidate_for_natural_speech_or_correct_terms(self, text: str) -> None:
        assert _find(text) == []

    @pytest.mark.parametrize("text, heard, replacement", [
        ("What we've seen in the burn is quite a clear downtrend.", "burn", "Bund"),
        ("Certainly in terms of pricing for the burns.", "burns", "Bunds"),
        ("This is where Mario Draggy can use the questions.", "Mario Draggy", "Mario Draghi"),
        ("We're looking at an E C B press conference.", "E C B", "ECB"),
        ("Details of the new T L T R O programme.", "T L T R O", "TLTRO"),
        ("The for word guidance was left unchanged.", "for word guidance", "Forward Guidance"),
        ("It was a slightly more hawk ish decision.", "hawk ish", "hawkish"),
        ("Currently suffering from a cute angina.", "a cute angina", "acute angina"),
        ("The statement was released from Frank Fort today.", "Frank Fort", "Frankfurt"),
    ])
    def test_finds_the_misheard_span_and_its_term(self, text: str, heard: str, replacement: str) -> None:
        candidates = _find(text)

        assert [(c.heard, c.replacement) for c in candidates] == [(heard, replacement)]
        assert text[candidates[0].start:candidates[0].end] == heard

    def test_confidently_heard_words_are_not_candidates(self) -> None:
        text = "What we've seen in the burn is a downtrend."

        assert _find(text, _words(text, {"burn": 0.97})) == []

    def test_low_probability_is_recorded_on_the_candidate(self) -> None:
        text = "What we've seen in the burn is a downtrend."

        candidates = _find(text, _words(text, {"burn": 0.18}))

        assert [(c.heard, c.min_probability) for c in candidates] == [("burn", 0.18)]

    def test_unmappable_words_do_not_block_a_candidate(self) -> None:
        text = "What we've seen in the burn is a downtrend."
        words = _words("Completely different words here", {})

        assert [c.heard for c in _find(text, words, max_word_probability=0.5)] == ["burn"]

    def test_plural_heard_for_a_singular_term_is_not_guessed(self) -> None:
        assert _find("They cut the depot sit rates again.") == []

    @pytest.mark.parametrize("text", [
        "The bond market burned through the lows.",
        "The market is burning through the lows.",
    ])
    def test_inflected_verb_is_not_turned_into_a_noun_term(self, text: str) -> None:
        # "bond" may still be a candidate (the verifier judges it); the verb form never is
        assert not any(c.heard.startswith("burn") for c in _find(text))

    def test_sentence_initial_lowercase_term_is_capitalised(self) -> None:
        assert [c.replacement for c in _find("hawk ish comments followed.")] == ["Hawkish"]

    def test_candidates_are_non_overlapping_and_in_text_order(self) -> None:
        candidates = _find("Mario Draggy said the for word guidance holds.")

        assert [c.heard for c in candidates] == ["Mario Draggy", "for word guidance"]
        assert candidates[0].end <= candidates[1].start

    def test_no_terms_or_empty_text(self) -> None:
        assert find_term_candidates("in the burn", [], [], 0.7, 0.6, 1.67, 0.95) == []
        assert _find("") == []


class TestSerbianCandidates:
    def test_misheard_serbian_term_with_diacritics_is_a_candidate(self, appliance_language):
        appliance_language("sr")

        candidates = find_term_candidates("Snimak pokazuje akutni infarkd miokarda.", [], ["infarkt miokarda"],
                                          0.7, 0.6, 1.67, 0.95)

        assert [(c.heard, c.replacement) for c in candidates] == [("infarkd miokarda", "infarkt miokarda")]

    def test_words_with_serbian_letters_are_whole_tokens(self, appliance_language):
        """With an a-z tokenizer "Šećer" split into "e" and "er" and could never match a term."""
        appliance_language("sr")

        candidates = find_term_candidates("Nivo šećera je visok.", [], ["šećer"], 0.7, 0.6, 1.67, 0.95)

        assert all(c.heard != "e" for c in candidates)

    def test_inflected_form_is_not_treated_as_a_plural(self, appliance_language):
        """Serbian has no "-s" plural: the English singularisation must never strip a Serbian word's ending."""
        appliance_language("sr")

        candidates = find_term_candidates("Uzeli smo tri stenta.", [], ["stents"], 0.7, 0.6, 1.67, 0.95)

        assert all(c.replacement == "stents" for c in candidates)


class TestSoundKeyCache:
    def test_a_language_switch_does_not_serve_the_previous_languages_keys(self, appliance_language):
        finder._cached_key_similarity.cache_clear()
        with patch.object(finder, "phonetic_key", side_effect=["abcd", "wxyz"]) as key:
            appliance_language("en")
            english = finder._key_similarity("heard", "abcd")
            appliance_language("sr")
            serbian = finder._key_similarity("heard", "abcd")
        finder._cached_key_similarity.cache_clear()

        assert (english, serbian) == (1.0, 0.0)
        assert key.call_count == 2
