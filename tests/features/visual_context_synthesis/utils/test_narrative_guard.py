# Path: tests/features/visual_context_synthesis/utils/test_narrative_guard.py

import pytest

from app.features.visual_context_synthesis.utils.narrative_guard import fallback_narrative, narrative_is_supported

SOURCES = [
    'Trading chart with on-screen text "TRADING DRILL"; "Target 161.17 on break of trend 161.84".',
    'Slide with on-screen text "BoE REVERSAL 22/3/18"; "Result 7-2 for hike BUT". Speaker\'s reference: the 7-2 vote.',
]


class TestNarrativeIsSupported:
    @pytest.mark.parametrize("narrative", [
        'A trading chart titled "TRADING DRILL" marks a 161.17 target, followed by the "BoE REVERSAL 22/3/18" slide.',
        "A chart and a slide about a 7-2 vote were shown.",
        "Charts were shown.",
    ])
    def test_numbers_and_quotes_from_the_descriptions_are_supported(self, narrative: str) -> None:
        assert narrative_is_supported(narrative, SOURCES) is True

    @pytest.mark.parametrize("narrative", [
        # A value no frame description contains
        "A trading chart shows the downtrend at 160.84.",
        # Loosely requoted on-screen text
        'A slide reads "Result: 7-2 vote for a hike".',
    ])
    def test_added_numbers_or_quotes_are_unsupported(self, narrative: str) -> None:
        assert narrative_is_supported(narrative, SOURCES) is False


class TestFallbackNarrative:
    def test_joins_distinct_descriptions_in_order(self) -> None:
        assert fallback_narrative(["A.", "B.", "A."]) == "A. B."

    def test_cuts_at_the_last_whole_sentence(self) -> None:
        assert fallback_narrative(["First sentence. Second sentence is long."], max_chars=25) == "First sentence."

    def test_cuts_mid_sentence_when_no_sentence_fits(self) -> None:
        assert fallback_narrative(["One very long sentence without a stop"], max_chars=8) == "One very…"
