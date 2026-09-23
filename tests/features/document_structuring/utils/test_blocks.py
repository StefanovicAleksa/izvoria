# Path: tests/features/document_structuring/utils/test_blocks.py

from app.features.document_structuring.domain.enums import BlockKind
from app.features.document_structuring.utils.blocks import parse_blocks
from app.features.document_structuring.utils.headings import compile_clause_patterns

PATTERNS = compile_clause_patterns([r"^(?:Article|ARTICLE)\s+[0-9]+"])


def _kinds(blocks):
    return [(b.kind, b.text) for b in blocks]


class TestParseBlocks:
    def test_heading_inside_an_ocr_paragraph_splits_it(self):
        text = "LEASE AGREEMENT\nArticle 7 Termination\nThe Tenant may terminate\nthis Agreement on notice."

        blocks = parse_blocks([(1, text)], PATTERNS)

        assert _kinds(blocks) == [
            (BlockKind.HEADING, "LEASE AGREEMENT"),
            (BlockKind.HEADING, "Article 7 Termination"),
            (BlockKind.PARAGRAPH, "The Tenant may terminate this Agreement on notice."),
        ]
        assert [b.level for b in blocks] == [1, 2, 0]

    def test_markdown_table_is_one_block(self):
        text = "Rent schedule:\n| Year | Rent |\n| --- | --- |\n| 2025 | 1,250 |\nPayable monthly."

        blocks = parse_blocks([(3, text)], PATTERNS)

        assert _kinds(blocks) == [
            (BlockKind.PARAGRAPH, "Rent schedule:"),
            (BlockKind.TABLE, "| Year | Rent |\n| --- | --- |\n| 2025 | 1,250 |"),
            (BlockKind.PARAGRAPH, "Payable monthly."),
        ]
        assert {b.page_no for b in blocks} == {3}

    def test_running_header_and_page_numbers_are_removed(self):
        pages = [(n, f"ACME PROPERTIES LTD - CONFIDENTIAL\nClause text on page {n}.\nPage {n} of 4") for n in range(1, 5)]

        blocks = parse_blocks(pages, PATTERNS)

        assert [b.text for b in blocks] == [f"Clause text on page {n}." for n in range(1, 5)]

    def test_numbers_in_the_body_are_kept(self):
        """A year in a table row is content, not a page number."""
        text = "Rent\n\n2025\n\nEUR 1,250.00\n\n2026\n\nEUR 1,300.00\n\nPayable in advance."

        blocks = parse_blocks([(3, text)], PATTERNS)

        assert "2025" in [b.text for b in blocks] and "2026" in [b.text for b in blocks]

    def test_a_bare_number_far_from_the_page_number_is_kept_even_at_the_bottom(self):
        blocks = parse_blocks([(3, "Total due\n1250")], PATTERNS)

        assert [b.text for b in blocks] == ["Total due 1250"]

    def test_repeated_body_sentence_is_not_page_furniture(self):
        body = "Top line {n}\nThe Tenant shall comply with the rules.\nMore text {n}\nEven more {n}\nEnd {n}"
        pages = [(n, body.format(n=n)) for n in range(1, 5)]

        blocks = parse_blocks(pages, PATTERNS)

        assert all("The Tenant shall comply with the rules." in b.text for b in blocks)

    def test_normalize_is_applied_to_every_block(self):
        blocks = parse_blocks([(1, "Article 1 Ugovor\nЗакупац плаћа.")], PATTERNS, normalize=str.upper)

        assert [b.text for b in blocks] == ["ARTICLE 1 UGOVOR", "ЗАКУПАЦ ПЛАЋА."]
