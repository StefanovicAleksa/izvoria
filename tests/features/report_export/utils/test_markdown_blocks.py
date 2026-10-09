# Path: tests/features/report_export/utils/test_markdown_blocks.py

from app.features.report_export.domain.entities import (
    Citation, Heading, InlineMath, LineBreak, ListBlock, MathBlock, Paragraph, Quote, Rule, TableBlock, TextRun,
)
from app.features.report_export.utils.markdown_blocks import parse, parse_inline


class TestParse:
    def test_a_formula_keeps_its_underscores_and_backslashes(self):
        """Unprotected, `$E_p$ … $Q_S$` would come back from the Markdown parser as italics."""
        (paragraph,) = parse("Elasticity $E_p$ and supply $Q_S$ and $\\bar{P}$.")

        assert [i.latex for i in paragraph.inlines if isinstance(i, InlineMath)] == ["E_p", "Q_S", "\\bar{P}"]
        assert not any(isinstance(i, TextRun) and i.italic for i in paragraph.inlines)

    def test_citations_become_citations_even_when_adjacent(self):
        (paragraph,) = parse("Shown twice [Doc 47][Doc 48].")

        assert paragraph.inlines == (TextRun("Shown twice "), Citation(47), Citation(48), TextRun("."))

    def test_a_bold_lead_in_keeps_its_weight(self):
        (paragraph,) = parse("**Substitutes:** a rise in one [Doc 45].")

        assert paragraph.inlines[0] == TextRun("Substitutes:", bold=True)
        assert paragraph.inlines[2] == Citation(45)

    def test_a_display_formula_alone_is_its_own_block(self):
        blocks = parse("Before.\n\n$$\\frac{\\Delta Q}{Q}$$\n\nAfter.")

        assert blocks[1] == MathBlock(latex="\\frac{\\Delta Q}{Q}")

    def test_money_stays_text(self):
        (paragraph,) = parse("From $6.40 to $3.00 per mcf.")

        assert paragraph.inlines == (TextRun("From $6.40 to $3.00 per mcf."),)

    def test_headings_lists_quotes_and_rules(self):
        blocks = parse("### Determinants\n\n- one\n- two [Doc 3]\n\n2. second\n3. third\n\n> quoted\n\n---")

        assert blocks[0] == Heading(level=3, inlines=(TextRun("Determinants"),))
        assert isinstance(blocks[1], ListBlock) and not blocks[1].ordered and len(blocks[1].items) == 2
        assert blocks[1].items[1] == (Paragraph(inlines=(TextRun("two "), Citation(3))),)
        assert isinstance(blocks[2], ListBlock) and blocks[2].ordered and blocks[2].start == 2
        assert blocks[3] == Quote(blocks=(Paragraph(inlines=(TextRun("quoted"),)),))
        assert blocks[4] == Rule()

    def test_a_table_with_math_and_citations_in_its_cells(self):
        (table,) = parse("| Good | Elasticity |\n|---|---|\n| Gas | $E_p = -0.2$ [Doc 3] |")

        assert isinstance(table, TableBlock)
        assert table.header == ((TextRun("Good"),), (TextRun("Elasticity"),))
        assert table.rows == (((TextRun("Gas"),), (InlineMath("E_p = -0.2"), TextRun(" "), Citation(3))),)

    def test_a_hard_break_is_kept_and_a_soft_one_is_a_space(self):
        (paragraph,) = parse("one\ntwo  \nthree")

        assert paragraph.inlines == (TextRun("one two"), LineBreak(), TextRun("three"))

    def test_code_shows_its_formula_as_written(self):
        (paragraph,) = parse("```\nE = $mc^2$\n```")

        assert paragraph.inlines == (TextRun("E = $mc^2$", code=True),)


class TestParseInline:
    def test_a_pack_list_item_comes_back_as_its_inlines(self):
        inlines = parse_inline("- **Storage** — no evidence supported this.")

        assert inlines[0] == TextRun("Storage", bold=True)

    def test_an_italic_note(self):
        assert parse_inline("_unresolved_") == (TextRun("unresolved", italic=True),)
