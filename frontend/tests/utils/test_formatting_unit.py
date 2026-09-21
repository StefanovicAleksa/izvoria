# Canvas: desktop_test_formatting_v1
# Path: frontend/tests/utils/test_formatting_unit.py

import pytest

from onyx_desktop.utils.formatting import (
    duration,
    elide,
    score,
    speaker_summary,
    split_doc_tags,
    timespan,
    timestamp,
)


class TestTimestamp:
    @pytest.mark.parametrize(
        "seconds, expected",
        [
            (0, "00:00"),
            (7.4, "00:07"),
            (72, "01:12"),
            (599.9, "09:59"),
            (3600, "1:00:00"),
            (3723, "1:02:03"),
            (-5, "00:00"),          # clamped: a negative timestamp is never displayed
            (None, "00:00"),        # tolerated: a missing value must not crash a card
        ],
    )
    def test_formats_seconds(self, seconds, expected):
        assert timestamp(seconds) == expected

    def test_timespan_uses_an_en_dash(self):
        assert timespan(72, 108) == "01:12–01:48"


class TestDuration:
    @pytest.mark.parametrize(
        "seconds, expected",
        [(0, "0s"), (45, "45s"), (60, "1m 00s"), (252, "4m 12s"), (3600, "1h 00m"), (3900, "1h 05m")],
    )
    def test_formats_wall_clock(self, seconds, expected):
        assert duration(seconds) == expected


class TestScore:
    def test_missing_score_renders_as_a_dash(self):
        # A hydration read returns no rerank score; showing 0.000 would imply it was ranked last.
        assert score(None) == "—"

    def test_present_score_is_three_decimals(self):
        assert score(0.8123456) == "0.812"


class TestSpeakerSummary:
    @pytest.mark.parametrize(
        "speakers, expected",
        [
            ((), ""),
            (("SPEAKER_01",), "SPEAKER_01"),
            (("SPEAKER_01", "SPEAKER_02"), "SPEAKER_01, SPEAKER_02"),
            (("A", "B", "C", "D"), "A, B +2"),
            (("A", "A", "B"), "A, B"),          # duplicates collapse
            (("", "A"), "A"),                    # blanks dropped
        ],
    )
    def test_summarizes(self, speakers, expected):
        assert speaker_summary(speakers) == expected


class TestSplitDocTags:
    def test_splits_prose_from_citation_tags(self):
        parts = split_doc_tags("It was signed [Doc 1] on the fourth [Doc 2].")
        assert parts == [
            ("It was signed ", None),
            ("[Doc 1]", 1),
            (" on the fourth ", None),
            ("[Doc 2]", 2),
            (".", None),
        ]

    def test_text_with_no_tags_is_one_plain_run(self):
        assert split_doc_tags("No citations here.") == [("No citations here.", None)]

    def test_adjacent_tags_are_separate_runs(self):
        # normalize_citation_tags rewrites "[Doc 1, 3]" into adjacent tags, so this is the shape
        # the gateway actually hands over.
        assert split_doc_tags("[Doc 1][Doc 3]") == [("[Doc 1]", 1), ("[Doc 3]", 3)]

    def test_a_tag_at_the_very_start_is_handled(self):
        assert split_doc_tags("[Doc 4] began it.") == [("[Doc 4]", 4), (" began it.", None)]

    def test_multi_digit_numbers_parse(self):
        assert split_doc_tags("[Doc 12]") == [("[Doc 12]", 12)]

    def test_reassembling_the_runs_returns_the_original(self):
        text = "A [Doc 1] b [Doc 22] c."
        assert "".join(segment for segment, _ in split_doc_tags(text)) == text


class TestElide:
    def test_collapses_whitespace(self):
        assert elide("a   b\n c") == "a b c"

    def test_truncates_with_an_ellipsis(self):
        assert elide("x" * 20, limit=10) == "x" * 9 + "…"

    def test_short_text_is_untouched(self):
        assert elide("short", limit=10) == "short"
