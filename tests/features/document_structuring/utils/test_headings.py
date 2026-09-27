# Path: tests/features/document_structuring/utils/test_headings.py

import pytest

from app.features.document_structuring.utils.headings import compile_clause_patterns, heading_level


@pytest.fixture
def english(language_pack):
    return compile_clause_patterns(language_pack("en").documents.clause_patterns)


@pytest.fixture
def serbian(language_pack):
    return compile_clause_patterns(language_pack("sr").documents.clause_patterns)


@pytest.fixture
def slovenian(language_pack):
    return compile_clause_patterns(language_pack("sl").documents.clause_patterns)


class TestHeadingLevel:
    @pytest.mark.parametrize("line, expected", [
        ("# Memorandum", (1, "Memorandum")),
        ("### Payment terms", (3, "Payment terms")),
        ("Article 7 Termination", (2, "Article 7 Termination")),
        ("ARTICLE IV", (2, "ARTICLE IV")),
        ("Schedule 2", (2, "Schedule 2")),
        ("4 Payment", (2, "4 Payment")),
        ("4.2 Late Payment", (3, "4.2 Late Payment")),
        ("4.2.1 Interest", (4, "4.2.1 Interest")),
        ("COMMERCIAL LEASE AGREEMENT", (1, "COMMERCIAL LEASE AGREEMENT")),
    ])
    def test_recognises_headings(self, english, line, expected):
        assert heading_level(line, english) == expected

    @pytest.mark.parametrize("line", [
        "",
        "The Tenant shall pay rent monthly.",
        "7.1 The Tenant may terminate this Agreement by giving the Landlord not less than thirty days notice.",
        "Article 7 applies to every lease signed after the Effective Date.",
        "| Period | Rent |",
        "EUR 1,250.00",
        "2025",
        "OK",
    ])
    def test_body_lines_are_not_headings(self, english, line):
        assert heading_level(line, english) is None

    @pytest.mark.parametrize("line", ["3. člen", "3. ČLEN", "Člen 12", "Priloga 1"])
    def test_slovenian_clause_markers(self, slovenian, line):
        assert heading_level(line, slovenian) == (2, line)


class TestBookChapters:
    @pytest.mark.parametrize("line", [
        "CHAPTER I", "CHAPTER IV.", "Chapter 12", "Chapter Twelve", "CHAPTER XII — The Storm", "Book II: Winter",
        "PART 3", "Prologue", "EPILOGUE",
    ])
    def test_english_chapters_start_a_clause_level_section(self, english, line):
        assert heading_level(line, english) == (2, line)

    @pytest.mark.parametrize("line", [
        "Chapter 3 of the Act provides that the",
        "Part of the reason he left was",
        "Book I read last summer and",
    ])
    def test_wrapped_prose_that_starts_like_a_chapter_is_not_a_heading(self, english, line):
        assert heading_level(line, english) is None

    @pytest.mark.parametrize("line", ["GLAVA II", "Glava 3", "Поглавље 4. Олуја", "Deo I", "Uvod", "Пролог"])
    def test_serbian_chapters_in_both_scripts(self, serbian, line):
        assert heading_level(line, serbian) == (2, line)

    def test_serbian_wrapped_prose_is_not_a_heading(self, serbian):
        assert heading_level("Glava 3 zakona propisuje da se", serbian) is None

    @pytest.mark.parametrize("line", ["Poglavje 4", "4. poglavje", "Del II: Zima", "Predgovor"])
    def test_slovenian_chapters(self, slovenian, line):
        assert heading_level(line, slovenian) == (2, line)


class TestShoutedLinesInBooks:
    """Found ingesting Project Gutenberg's Sherlock Holmes: dialogue in capitals became chapter titles."""

    @pytest.mark.parametrize("line", [
        "“BALLARAT.”", "“‘ROBERT ST. SIMON.’", "“HUNTER.”", "„POMOČ!“", "\"STOP!\"",
        "DAMAGE.", "LIMITED TO WARRANTIES OF MERCHANTABILITY OR FITNESS FOR ANY PURPOSE.", "WHO IS THERE?",
    ])
    def test_quoted_or_finished_sentences_in_capitals_are_not_headings(self, english, line):
        assert heading_level(line, english) is None

    @pytest.mark.parametrize("line, expected", [
        ("I. A SCANDAL IN BOHEMIA", (1, "I. A SCANDAL IN BOHEMIA")),
        ("RECITALS:", (1, "RECITALS:")),
        ("SCHEDULE OF RENT", (1, "SCHEDULE OF RENT")),
    ])
    def test_titles_in_capitals_are_still_headings(self, english, line, expected):
        assert heading_level(line, english) == expected
