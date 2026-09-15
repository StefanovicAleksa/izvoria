# Path: tests/features/visual_analysis/utils/test_grounding_check.py

import pytest

from app.features.visual_analysis.domain.entities import FrameObservation
from app.features.visual_analysis.utils.grounding_check import (
    drop_items_already_reported,
    drop_marks_copied_from_other_frames,
    ground_observation
)

TRANSCRIPT = ("We've got a clear downtrend which comes in at 160.84 and that comes in if we can get back above 63. "
              "We'll have a look at the position squeeze.")


class TestGroundObservation:
    @pytest.mark.parametrize("observation", [
        None,
        FrameObservation(shows="speaker only"),
        FrameObservation(shows="Talking head of the presenter", legible_text=["AXIA"]),
        FrameObservation(shows="", legible_text=["Q3"]),
        # Nothing legible and nothing specific marked
        FrameObservation(shows="trading ladder"),
        FrameObservation(shows="trading ladder", marks="arrows pointing to different parts of the ladder"),
        FrameObservation(shows="trading ladder", marks="box pointing to chart"),
        FrameObservation(shows="trading ladder", marks="arrow pointing to a specific point on the ladder"),
        FrameObservation(shows="trading ladder", marks="blue arrows pointing to lines on the chart"),
        # Only a watermark and a speech copy remain to be removed
        FrameObservation(shows="chart", legible_text=["www.fttuts.com", "160.84 if we can get back above 63"]),
    ])
    def test_frames_without_visible_evidence_get_no_description(self, observation) -> None:
        assert ground_observation(observation, TRANSCRIPT) is None

    def test_removes_websites_and_speech_copies_but_keeps_read_text(self) -> None:
        observation = FrameObservation(
            shows="chart",
            legible_text=["TRADING DRILL", "www.fttuts.com", "160.84 if we can get back above 63",
                          "Target 161.17 on break of trend 161.84", "POSITION SQUEEZE"],
            marks="arrows pointing to different parts of the chart"
        )

        grounded = ground_observation(observation, TRANSCRIPT)

        # A title the speaker also says has no number, so it stays
        assert grounded.legible_text == ["TRADING DRILL", "Target 161.17 on break of trend 161.84", "POSITION SQUEEZE"]
        assert grounded.marks is None

    @pytest.mark.parametrize("marks", [
        "doctor's finger pointing at a shadow in the lower lobe",
        "blue arrow pointing to wedge break; box pointing to chart",
        "red circle around the signature",
    ])
    def test_a_mark_with_a_specific_target_is_evidence(self, marks: str) -> None:
        observation = FrameObservation(shows="chest X-ray", marks=marks)

        assert ground_observation(observation, TRANSCRIPT).marks == marks

    def test_specific_marks_alone_are_evidence(self) -> None:
        observation = FrameObservation(shows="candlestick chart", marks="horizontal line at the break of structure")

        assert ground_observation(observation, TRANSCRIPT) == observation

    def test_speech_link_with_a_number_the_speaker_never_said_is_dropped(self) -> None:
        observation = FrameObservation(shows="chart", legible_text=["TRADING DRILL"], speech_link="the move to 142.27")

        assert ground_observation(observation, TRANSCRIPT).speech_link is None

    def test_speech_link_repeating_the_speech_is_kept(self) -> None:
        observation = FrameObservation(shows="chart", legible_text=["TRADING DRILL"], speech_link="the 160.84 downtrend")

        assert ground_observation(observation, TRANSCRIPT).speech_link == "the 160.84 downtrend"

    def test_removes_player_chrome_and_repeated_items(self) -> None:
        observation = FrameObservation(
            shows="slide", legible_text=["BoE REVERSAL 2/8/18", "16 / 31", "00:21 / 04:32", "PREV", "Marker Tools",
                                         "boe reversal 2/8/18", "Exit: exhaustion down to 1.3060"]
        )

        assert ground_observation(observation, TRANSCRIPT).legible_text == [
            "BoE REVERSAL 2/8/18", "Exit: exhaustion down to 1.3060"
        ]

    def test_removes_an_outline_sidebar_but_keeps_long_numbered_bullets(self) -> None:
        observation = FrameObservation(
            shows="slide", legible_text=["BoE REVERSAL 2/8/18", "1. Fade trade: fast recognition of range 1.3150-20",
                                         "15. BOE Reversal 2/8/18", "16. BOE Reversal 2/8/18", "18. Continuation"]
        )

        assert ground_observation(observation, TRANSCRIPT).legible_text == [
            "BoE REVERSAL 2/8/18", "1. Fade trade: fast recognition of range 1.3150-20"
        ]

    def test_two_short_numbered_items_are_not_an_outline(self) -> None:
        observation = FrameObservation(shows="slide", legible_text=["1. Fade trade", "2. Breakout"])

        assert ground_observation(observation, TRANSCRIPT).legible_text == ["1. Fade trade", "2. Breakout"]

    def test_keeps_at_most_eight_items(self) -> None:
        observation = FrameObservation(shows="slide", legible_text=[f"Bullet point number {c}" for c in "ABCDEFGHIJ"])

        assert len(ground_observation(observation, TRANSCRIPT).legible_text) == 8

    def test_short_item_holding_a_spoken_number_is_a_speech_copy(self) -> None:
        observation = FrameObservation(shows="chart", legible_text=["TRADING DRILL", "160.84 downtrend"])

        assert ground_observation(observation, TRANSCRIPT).legible_text == ["TRADING DRILL"]

    def test_a_spoken_sentence_reported_as_text_is_a_speech_copy(self) -> None:
        transcript = "As you can see right here, there appears to be a shadow in the lower lobe."
        observation = FrameObservation(shows="X-ray image", legible_text=["There appears to be a shadow in the lower lobe."],
                                       speech_link="There appears to be a shadow in the lower lobe.")

        assert ground_observation(observation, transcript) is None

    def test_words_of_the_speech_scattered_across_a_bullet_are_not_a_copy(self) -> None:
        observation = FrameObservation(shows="slide", legible_text=["Market pre-positioned for a technical move"])
        transcript = "The market had moved. We will look at a technical setup. People positioned early."

        assert ground_observation(observation, transcript).legible_text == ["Market pre-positioned for a technical move"]

    def test_marks_naming_a_spoken_number_the_frame_text_does_not_show_are_dropped(self) -> None:
        spoken = FrameObservation(shows="ladder", legible_text=["TRADING DRILL"], marks='arrow pointing to "160.84"')
        legible = FrameObservation(shows="chart", legible_text=["Downtrend 160.84 line"], marks="arrow at 160.84")

        assert ground_observation(spoken, TRANSCRIPT).marks is None
        assert ground_observation(legible, TRANSCRIPT).marks == "arrow at 160.84"


class TestDropItemsAlreadyReported:
    def test_removes_items_an_earlier_frame_reported_and_records_new_ones(self) -> None:
        reported = {"boe reversal 2/8/18"}
        observation = FrameObservation(shows="ladder", legible_text=["BoE REVERSAL 2/8/18", "Stops hit instantly"])

        result = drop_items_already_reported(observation, reported)

        assert result.legible_text == ["Stops hit instantly"]
        assert reported == {"boe reversal 2/8/18", "stops hit instantly"}

    def test_a_frame_with_nothing_new_and_no_marks_is_dropped(self) -> None:
        observation = FrameObservation(shows="ladder", legible_text=["BoE REVERSAL 2/8/18"])

        assert drop_items_already_reported(observation, {"boe reversal 2/8/18"}) is None

    def test_marks_keep_a_frame_whose_text_was_already_reported(self) -> None:
        observation = FrameObservation(shows="chart", legible_text=["BoE REVERSAL 2/8/18"], marks="arrow at the spike")

        assert drop_items_already_reported(observation, {"boe reversal 2/8/18"}).legible_text == []


class TestDropMarksCopiedFromOtherFrames:
    def test_marks_pointing_at_another_frames_text_are_cleared(self) -> None:
        slide = FrameObservation(shows="slide", legible_text=["BoE REVERSAL 22/3/18"],
                                 marks="blue arrow pointing to spike up on no major change")
        chart = FrameObservation(shows="chart", legible_text=["Spike up on no major change"],
                                 marks="blue arrow pointing to spike up on no major change")

        assert drop_marks_copied_from_other_frames([slide, chart, None]) == [
            slide.model_copy(update={"marks": None}), chart, None
        ]

    def test_a_frame_left_without_evidence_becomes_none(self) -> None:
        ladder = FrameObservation(shows="ladder", marks="arrow pointing to fade after top of wedge hit")
        chart = FrameObservation(shows="chart", legible_text=["Fade after top of wedge hit"])

        assert drop_marks_copied_from_other_frames([ladder, chart]) == [None, chart]

    def test_marks_about_the_frames_own_text_or_unrelated_things_are_kept(self) -> None:
        own = FrameObservation(shows="slide", legible_text=["9-0 hawkish: spike up squeeze"],
                               marks='blue arrow pointing to "9-0 hawkish: spike up squeeze"')
        unrelated = FrameObservation(shows="chart", legible_text=["Wedge break"], marks="horizontal line at the break")

        assert drop_marks_copied_from_other_frames([own, unrelated]) == [own, unrelated]
