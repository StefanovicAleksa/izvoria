# Path: tests/features/visual_analysis/utils/test_frame_description_parser.py

from app.features.visual_analysis.utils.frame_description_parser import parse_frame_descriptions


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
