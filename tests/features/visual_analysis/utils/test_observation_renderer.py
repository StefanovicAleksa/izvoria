# Path: tests/features/visual_analysis/utils/test_observation_renderer.py

from app.features.visual_analysis.domain.entities import FrameObservation
from app.features.visual_analysis.utils.observation_renderer import render_observation


class TestRenderObservation:
    def test_screen_marks_and_speech_each_get_their_own_sentence(self):
        observation = FrameObservation(
            shows="slide", legible_text=["BoE REVERSAL 22/3/18", "Result 7-2 for hike BUT"],
            marks="arrow pointing at the spike.", speech_link="the 7-2 vote"
        )

        assert render_observation(observation) == (
            'Slide with on-screen text "BoE REVERSAL 22/3/18"; "Result 7-2 for hike BUT". '
            "Marked on screen: arrow pointing at the spike. "
            "Speaker's reference: the 7-2 vote."
        )

    def test_marks_without_legible_text(self):
        observation = FrameObservation(shows="candlestick chart", marks="horizontal line at the break of structure",
                                       speech_link="the break of structure at 131.19")

        assert render_observation(observation) == (
            "Candlestick chart. Marked on screen: horizontal line at the break of structure. "
            "Speaker's reference: the break of structure at 131.19."
        )

    def test_shows_ending_in_with_text_is_not_repeated(self):
        observation = FrameObservation(shows="Slide with text", legible_text=["POSITION SQUEEZE"])

        assert render_observation(observation) == 'Slide with on-screen text "POSITION SQUEEZE".'
