# Path: tests/features/visual_hallucination_correction/utils/test_topic_overlap.py

import pytest

from app.features.visual_hallucination_correction.utils.topic_overlap import shared_on_screen_topic, shares_topic

TRANSCRIPT = ("So what you can see here in the pound is that we've had quite a rally going into the Bank of England "
              "decision, focusing on a 7-2 vote for a rate hike and a possible reversal.")


class TestSharedOnScreenTopic:
    def test_only_quoted_on_screen_text_counts(self) -> None:
        descriptions = ['Slide with on-screen text "BoE REVERSAL 22/3/18"; "Result 7-2 for hike BUT". '
                        "Speaker's reference: the rally going into the decision."]

        assert shared_on_screen_topic(TRANSCRIPT, descriptions) == {"reversal", "hike", "7-2"}


class TestSharesTopic:
    @pytest.mark.parametrize("descriptions", [
        ['Slide with on-screen text "Expectation: rate hike"; "Reversal trade".'],
        # One specific number is enough
        ['Chart with on-screen text "Result 7-2".'],
    ])
    def test_on_screen_text_that_belongs_to_the_speech(self, descriptions) -> None:
        assert shares_topic(TRANSCRIPT, descriptions, min_shared=2) is True

    @pytest.mark.parametrize("descriptions", [
        ['X-ray with on-screen text "PA CHEST"; "R". Marked on screen: arrow at a shadow in the lower left lung.'],
        ['Document with on-screen text "SUPERIOR COURT OF THE STATE OF CALIFORNIA"; "MOTION TO DISMISS".'],
        # One shared ordinary word, or shared words outside the quotes, isn't enough
        ['Photo with on-screen text "TOMATO SOUP"; "Decision time". Speaker\'s reference: the pound rally.'],
        ["Trading chart. Marked on screen: arrow at the rally."],
    ])
    def test_unrelated_or_unquoted_evidence_goes_to_the_checker(self, descriptions) -> None:
        assert shares_topic(TRANSCRIPT, descriptions, min_shared=2) is False
