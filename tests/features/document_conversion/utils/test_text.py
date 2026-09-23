# Path: tests/features/document_conversion/utils/test_text.py

import pytest

from app.features.document_conversion.utils.text import count_letters, normalize_extracted_text


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
