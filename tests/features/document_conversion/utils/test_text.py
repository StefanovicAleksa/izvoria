# Path: tests/features/document_conversion/utils/test_text.py

import pytest

from app.features.document_conversion.utils.text import count_letters, normalize_extracted_text, normalize_text_lines


class TestCountLetters:
    @pytest.mark.parametrize("text, expected", [
        ("", 0), ("| 12 | 3.50 |", 0), ("Član 7", 4), ("Члан 7", 4), ("abc 123 def", 6),
    ])
    def test_counts_letters_in_any_script(self, text, expected):
        assert count_letters(text) == expected


class TestNormalizeExtractedText:
    @pytest.mark.parametrize("raw, expected", [
        ("Deﬁnitions and oﬃce", "Definitions and office"),
        ("word\tword\t\tword", "word word word"),
        ("non breaking", "non breaking"),
        ("hyph­enated", "hyphenated"),
        ("  trailing   \n  leading", "trailing\nleading"),
        ("para one\n\n\n\n\npara two", "para one\n\npara two"),
        ("line\r\nbreak", "line\nbreak"),
    ])
    def test_changes_whitespace_and_ligatures_only(self, raw, expected):
        assert normalize_extracted_text(raw) == expected

    def test_keeps_diacritics(self):
        assert normalize_extracted_text("Najemnik plača za čas najema") == "Najemnik plača za čas najema"


class TestNormalizeTextLines:
    def test_keeps_every_line_including_leading_and_blank_ones(self):
        raw = "\n\nCHAPTER I\n\n\n\nIt was night.\n"

        assert normalize_text_lines(raw).split("\n") == ["", "", "CHAPTER I", "", "", "", "It was night.", ""]

    @pytest.mark.parametrize("raw, expected", [
        ("Deﬁnitions", "Definitions"),
        ("  word\tword  ", "word word"),
        ("one\r\ntwo\rthree", "one\ntwo\nthree"),
        ("hyph­enated", "hyphenated"),
    ])
    def test_cleans_inside_lines_only(self, raw, expected):
        assert normalize_text_lines(raw) == expected
