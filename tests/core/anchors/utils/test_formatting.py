# Path: tests/core/anchors/utils/test_formatting.py

import pytest

from app.core.anchors.domain.enums import AnchorKind
from app.core.anchors.utils.formatting import format_anchor, format_location, format_pages, format_timestamp


class TestFormatTimestamp:
    @pytest.mark.parametrize("seconds, expected", [
        (0, "00:00:00"), (59.9, "00:00:59"), (751, "00:12:31"), (3725, "01:02:05"), (-3, "00:00:00"),
    ])
    def test_formats_hours_minutes_seconds(self, seconds, expected):
        assert format_timestamp(seconds) == expected


class TestFormatAnchor:
    @pytest.mark.parametrize("code, kind, kwargs, expected", [
        ("en", AnchorKind.TIME, {"start_sec": 751, "end_sec": 842}, "00:12:31–00:14:02"),
        ("en", AnchorKind.PAGE, {"page_start": 7, "page_end": 7}, "p. 7"),
        ("en", AnchorKind.PAGE, {"page_start": 7, "page_end": 8}, "pp. 7–8"),
        ("en", AnchorKind.PAGE, {"page_start": 7}, "p. 7"),
        ("sl", AnchorKind.PAGE, {"page_start": 3, "page_end": 4}, "str. 3–4"),
        ("sr", AnchorKind.PAGE, {"page_start": 3}, "str. 3"),
        ("en", AnchorKind.SECTION, {"section_path": "Terms › Payment"}, "Terms › Payment"),
        # A page anchor whose page is unknown falls back to its section rather than inventing a page
        ("en", AnchorKind.PAGE, {"section_path": "Article 7"}, "Article 7"),
        ("en", AnchorKind.SECTION, {}, ""),
        # A line anchor carries its lines in the page fields
        ("en", AnchorKind.LINE, {"page_start": 12, "page_end": 12}, "line 12"),
        ("en", AnchorKind.LINE, {"page_start": 120, "page_end": 164}, "lines 120–164"),
        ("sl", AnchorKind.LINE, {"page_start": 3, "page_end": 9}, "vrstice 3–9"),
        ("sr", AnchorKind.LINE, {"page_start": 3}, "red 3"),
        ("en", AnchorKind.LINE, {"section_path": "Chapter 4"}, "Chapter 4"),
        # An image is cited as a whole, in the pack's word for it
        ("en", AnchorKind.IMAGE, {}, "image"),
        ("sr", AnchorKind.IMAGE, {"section_path": "ignored"}, "slika"),
    ])
    def test_renders_each_kind_in_the_appliance_language(self, language_pack, code, kind, kwargs, expected):
        assert format_anchor(kind, rules=language_pack(code).documents, **kwargs) == expected

    def test_accepts_the_stored_string_value(self, language_pack):
        assert format_anchor("page", page_start=2, rules=language_pack("en").documents) == "p. 2"

    def test_uses_the_active_pack_when_none_is_given(self, appliance_language):
        appliance_language("sl")

        assert format_pages(5, 5) == "str. 5"


class TestFormatLocation:
    def test_paged_document_adds_its_section(self, language_pack):
        location = format_location(AnchorKind.PAGE, page_start=7, page_end=8, section_path="Article 7 › 7.2",
                                   rules=language_pack("en").documents)

        assert location == "pp. 7–8 · Article 7 › 7.2"

    def test_line_numbered_document_adds_its_section(self, language_pack):
        location = format_location(AnchorKind.LINE, page_start=120, page_end=164, section_path="CHAPTER IV",
                                   rules=language_pack("en").documents)

        assert location == "lines 120–164 · CHAPTER IV"

    def test_transcript_location_is_only_its_time_range(self, language_pack):
        location = format_location(AnchorKind.TIME, 10, 20, section_path="ignored",
                                   rules=language_pack("en").documents)

        assert location == "00:00:10–00:00:20"

    def test_section_anchor_is_not_repeated(self, language_pack):
        location = format_location(AnchorKind.SECTION, section_path="Terms", rules=language_pack("en").documents)

        assert location == "Terms"
