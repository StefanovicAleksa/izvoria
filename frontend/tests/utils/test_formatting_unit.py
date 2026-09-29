# Canvas: desktop_test_formatting_v1
# Path: frontend/tests/utils/test_formatting_unit.py

import uuid

import pytest

from onyx_desktop.utils.formatting import (
    duration,
    elide,
    score,
    speaker_summary,
    split_doc_tags,
    timespan,
    timestamp,
    when,
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


class TestLocation:
    def test_a_document_excerpt_shows_its_page(self):
        from onyx_desktop.domain.entities import Evidence
        from onyx_desktop.utils.formatting import location

        evidence = Evidence(doc_number=1, chunk_id=uuid.uuid4(), source_id=uuid.uuid4(), start_sec=0.0,
                            end_sec=0.0, location="p. 7 · Article 7")

        assert location(evidence) == "p. 7 · Article 7"

    def test_a_recording_excerpt_shows_its_time_range(self):
        from onyx_desktop.domain.entities import Evidence
        from onyx_desktop.utils.formatting import location

        evidence = Evidence(doc_number=1, chunk_id=uuid.uuid4(), source_id=uuid.uuid4(), start_sec=72, end_sec=108)

        assert location(evidence) == "01:12–01:48"


class TestWhen:
    """Local time in, so naive datetimes pin the expectations without depending on the machine's zone."""
    NOW = __import__("datetime").datetime(2026, 9, 28, 18, 0)

    @pytest.mark.parametrize("moment, expected", [
        ((2026, 9, 28, 14, 32), "14:32"),
        ((2026, 9, 27, 9, 5), "Yesterday 09:05"),
        ((2026, 9, 12, 14, 32), "12 Sep 14:32"),
        ((2025, 9, 12, 14, 32), "12 Sep 2025"),
    ])
    def test_as_short_as_stays_unambiguous(self, moment, expected):
        from datetime import datetime
        assert when(datetime(*moment), now=self.NOW) == expected

    def test_nothing_for_no_date(self):
        assert when(None) == ""

    def test_an_aware_datetime_is_shown_in_local_time(self):
        from datetime import datetime, timezone
        assert when(datetime.now(timezone.utc)) == datetime.now().astimezone().strftime("%H:%M")
