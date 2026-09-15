# Path: tests/features/visual_analysis/utils/test_frame_description_parser.py

from app.features.visual_analysis.domain.entities import FrameObservation
from app.features.visual_analysis.utils.frame_description_parser import (
    parse_frame_descriptions,
    parse_frame_observation,
    parse_frame_observations
)


class TestParseFrameDescriptions:
    def test_one_line_per_frame(self):
        assert parse_frame_descriptions("Frame 1: A slide.\nFrame 2: A chart.", 2) == ["A slide.", "A chart."]

    def test_tolerates_preamble_markdown_bullets_and_wrapped_lines(self):
        raw = "Here are the descriptions:\n**Frame 1:** A slide titled\n'Q3 Results'.\n- Frame 2 - A chart."

        assert parse_frame_descriptions(raw, 2) == ["A slide titled 'Q3 Results'.", "A chart."]

    def test_unmarked_answer_is_attributed_to_the_first_frame(self):
        assert parse_frame_descriptions("A slide and a chart.", 2) == ["A slide and a chart.", ""]

    def test_single_frame_without_marker(self):
        assert parse_frame_descriptions("  A slide.\n", 1) == ["A slide."]

    def test_out_of_range_and_empty_sections_are_ignored(self):
        raw = "Frame 1: A slide.\nFrame 3: An invented frame.\nFrame 2:"

        assert parse_frame_descriptions(raw, 2) == ["A slide.", ""]

    def test_repeated_marker_is_appended(self):
        assert parse_frame_descriptions("Frame 1: A slide.\nFrame 1: With a logo.", 1) == ["A slide. With a logo."]

    def test_zero_frames(self):
        assert parse_frame_descriptions("Frame 1: A slide.", 0) == []


class TestParseFrameObservation:
    def test_reads_every_field(self):
        section = ('SHOWS: slide. TEXT: "BoE REVERSAL 22/3/18" ; "Result 7-2 for hike BUT" '
                   'MARKS: arrow pointing at the spike. LINK: the 7-2 vote.')

        assert parse_frame_observation(section) == FrameObservation(
            shows="slide", legible_text=["BoE REVERSAL 22/3/18", "Result 7-2 for hike BUT"],
            marks="arrow pointing at the spike", speech_link="the 7-2 vote"
        )

    def test_none_values_and_markdown_labels(self):
        section = '**SHOWS:** chart **TEXT:** NONE **MARKS:** "None" **LINK:** N/A'

        assert parse_frame_observation(section) == FrameObservation(shows="chart", legible_text=[], marks=None,
                                                                    speech_link=None)

    def test_unquoted_text_is_split_on_semicolons(self):
        observation = parse_frame_observation("SHOWS: slide TEXT: Q3 Results; Revenue up 4% MARKS: NONE LINK: NONE")

        assert observation.legible_text == ["Q3 Results", "Revenue up 4%"]

    def test_lower_case_label_inside_a_value_does_not_start_a_field(self):
        observation = parse_frame_observation('SHOWS: document with a note text: unreadable TEXT: "Exhibit A"')

        assert observation.shows == "document with a note text: unreadable"
        assert observation.legible_text == ["Exhibit A"]

    def test_copied_format_placeholders_are_empty(self):
        dots = parse_frame_observation('SHOWS: photo TEXT: "..." ; "..." MARKS: ... LINK: ...')
        angle = parse_frame_observation("SHOWS: X-ray TEXT: <legible text items, or NONE> MARKS: <marks> LINK: NONE")

        assert dots == FrameObservation(shows="photo")
        assert angle == FrameObservation(shows="X-ray")
        assert parse_frame_observation("SHOWS: <what is on screen> TEXT: NONE") is None

    def test_section_without_shows_is_unstructured(self):
        assert parse_frame_observation("A slide showing quarterly revenue.") is None
        assert parse_frame_observation("SHOWS: NONE TEXT: \"x\"") is None


class TestParseFrameObservations:
    def test_one_observation_per_frame_and_none_for_unstructured_or_missing(self):
        raw = ("Frame 1:\nSHOWS: slide\nTEXT: \"Q3\"\nMARKS: NONE\nLINK: NONE\n"
               "Frame 2: A chart with no fields.\n")

        assert parse_frame_observations(raw, 3) == [
            FrameObservation(shows="slide", legible_text=["Q3"]), None, None
        ]
