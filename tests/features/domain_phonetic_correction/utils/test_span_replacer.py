from app.features.domain_phonetic_correction.domain.entities import PhoneticCandidate
from app.features.domain_phonetic_correction.utils.span_replacer import replace_spans


def _candidate(text: str, heard: str, replacement: str, occurrence: int = 0) -> PhoneticCandidate:
    start = -1
    for _ in range(occurrence + 1):
        start = text.index(heard, start + 1)
    return PhoneticCandidate(start=start, end=start + len(heard), heard=heard, term=replacement,
                             replacement=replacement, score=0.9)


class TestReplaceSpans:
    def test_replaces_only_the_span_and_keeps_punctuation(self) -> None:
        text = "What we've seen in the burn, is a downtrend."
        assert replace_spans(text, [_candidate(text, "burn", "Bund")]) == "What we've seen in the Bund, is a downtrend."

    def test_applies_several_spans_in_text_order(self) -> None:
        text = "Mario Draggy spoke about the for word guidance."
        candidates = [_candidate(text, "for word guidance", "Forward Guidance"), _candidate(text, "Mario Draggy", "Mario Draghi")]
        assert replace_spans(text, candidates) == "Mario Draghi spoke about the Forward Guidance."

    def test_replaces_the_exact_occurrence_not_the_first_match(self) -> None:
        text = "burn the burn"
        assert replace_spans(text, [_candidate(text, "burn", "Bund", occurrence=1)]) == "burn the Bund"

    def test_skips_a_candidate_whose_span_no_longer_holds_the_heard_text(self) -> None:
        text = "in the burn"
        moved = PhoneticCandidate(start=0, end=4, heard="burn", term="Bund", replacement="Bund", score=0.9)
        assert replace_spans(text, [moved]) == text

    def test_skips_an_overlapping_candidate(self) -> None:
        text = "hawk ish tone"
        first = PhoneticCandidate(start=0, end=8, heard="hawk ish", term="hawkish", replacement="hawkish", score=0.9)
        overlapping = PhoneticCandidate(start=5, end=13, heard="ish tone", term="x", replacement="x", score=0.8)
        assert replace_spans(text, [first, overlapping]) == "hawkish tone"

    def test_rejects_everything_if_digits_would_change(self) -> None:
        text = "those 73s"
        digit_eater = PhoneticCandidate(start=6, end=9, heard="73s", term="Bunds", replacement="Bunds", score=0.9)
        assert replace_spans(text, [digit_eater]) == text

    def test_no_candidates_returns_text_unchanged(self) -> None:
        assert replace_spans("unchanged text", []) == "unchanged text"
