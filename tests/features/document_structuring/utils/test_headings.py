# Path: tests/features/document_structuring/utils/test_headings.py

import pytest

from app.features.document_structuring.utils.headings import compile_clause_patterns, heading_level


@pytest.fixture
def english(language_pack):
    return compile_clause_patterns(language_pack("en").documents.clause_patterns)


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
