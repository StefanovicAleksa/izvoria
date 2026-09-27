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


class TestShoutedParagraphs:
    """A contract's liability disclaimer, hard-wrapped in capitals, is a paragraph -- not a run of headings."""

    def test_wrapped_capitals_paragraph_stays_one_paragraph_under_its_clause(self):
        text = ("Article 12 Limitation of Liability\n\n"
                "IN NO EVENT SHALL EITHER PARTY BE LIABLE FOR ANY INDIRECT,\n"
                "INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF\n"
                "OR IN CONNECTION WITH THIS AGREEMENT, EVEN IF ADVISED OF THE\n"
                "POSSIBILITY OF SUCH DAMAGES.")

        blocks = parse_blocks([(1, text)], PATTERNS)

        assert _kinds(blocks) == [
            (BlockKind.HEADING, "Article 12 Limitation of Liability"),
            (BlockKind.PARAGRAPH, "IN NO EVENT SHALL EITHER PARTY BE LIABLE FOR ANY INDIRECT, INCIDENTAL OR "
                                  "CONSEQUENTIAL DAMAGES ARISING OUT OF OR IN CONNECTION WITH THIS AGREEMENT, EVEN "
                                  "IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGES."),
        ]

    def test_capitals_title_after_a_finished_sentence_is_still_a_heading(self):
        text = "The parties agree as set out below.\nSCHEDULE OF RENT\nRent is due monthly."

        blocks = parse_blocks([(1, text)], PATTERNS)

        assert _kinds(blocks) == [
            (BlockKind.PARAGRAPH, "The parties agree as set out below."),
            (BlockKind.HEADING, "SCHEDULE OF RENT"),
            (BlockKind.PARAGRAPH, "Rent is due monthly."),
        ]

    def test_clause_heading_after_an_unfinished_line_still_splits(self):
        text = "The Tenant shall pay\nArticle 7 Termination\nOn notice."

        blocks = parse_blocks([(1, text)], PATTERNS)

        assert (BlockKind.HEADING, "Article 7 Termination") in _kinds(blocks)


class TestLineNumbers:
    def test_blocks_record_their_source_lines(self):
        text = "\n\nArticle 1 Terms\nThe rent is due\nevery month.\n\n\n| a | b |\n| 1 | 2 |\nLast line."

        blocks = parse_blocks([(None, text)], PATTERNS)

        assert [(b.kind, b.line_start, b.line_end) for b in blocks] == [
            (BlockKind.HEADING, 3, 3),
            (BlockKind.PARAGRAPH, 4, 5),
            (BlockKind.TABLE, 8, 9),
            (BlockKind.PARAGRAPH, 10, 10),
        ]

    def test_removed_page_furniture_does_not_renumber_the_lines_below_it(self):
        pages = [(n, f"ACME LTD CONFIDENTIAL\nBody of page {n}.\nMore body.") for n in range(1, 5)]

        blocks = parse_blocks(pages, PATTERNS)

        assert all((b.line_start, b.line_end) == (2, 3) for b in blocks)
