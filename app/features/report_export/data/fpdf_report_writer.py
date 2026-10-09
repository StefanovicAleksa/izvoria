# Path: app/features/report_export/data/fpdf_report_writer.py

"""
A report as a typeset PDF, laid out with fpdf2.

The content is the report as stored, the same as its Markdown file: the title, the provenance lines
(`markdown_renderer.provenance_lines`), the executive summary, each written section's own Markdown, what the corpus
could not support, and references for every cited `[Doc N]`. Nothing is asked of a model here.

What the PDF adds over the Markdown is reading: a title block, a running header and page numbers, PDF bookmarks per
section, citation tags that jump to their reference, typeset formulas, and references grouped under their source
with the start of each cited excerpt beside it, so a printed copy can be checked against its sources.

fpdf2 draws; the layout (where lines break, when a page turns) is this module's, because a sentence here holds
formulas and tags, which fpdf2's own text flow cannot put inside a line. All lengths are points.
"""

import re
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from fpdf import FPDF

from app.core.anchors.domain.enums import AnchorKind
from app.core.anchors.utils.formatting import format_location
from app.core.language.service.language_api import get_language_pack
from app.features.deep_research_report.domain.entities import EvidenceEntry, Report
from app.features.deep_research_report.utils.markdown_renderer import provenance_lines
from app.features.report_export.domain.entities import (
    Block, Citation, Heading, Inline, InlineMath, LineBreak, ListBlock, MathBlock, Paragraph, Quote, Rule,
    TableBlock, TextRun,
)
from app.features.report_export.domain.interfaces import IReportDocumentWriter
from app.features.report_export.exceptions.rendering import MathRenderError
from app.features.report_export.utils import fonts, markdown_blocks, math_render
from app.features.report_export.utils.line_layout import Atom, Line, break_lines

# ---------- the page ----------
PAGE_W, PAGE_H = 595.28, 841.89          # A4
MARGIN_X, MARGIN_TOP, MARGIN_BOTTOM = 64.0, 70.0, 66.0

# ---------- colour (the web app's light palette) ----------
TEXT = (31, 29, 26)
MUTED = (107, 102, 94)
FAINT = (150, 144, 134)
ACCENT = (45, 106, 88)
RULE = (228, 223, 214)
PANEL = (245, 242, 237)
NOTICE_PANEL = (251, 244, 228)
NOTICE_BAR = (176, 125, 32)

# ---------- type ----------
BODY = 10.5
LEADING = 1.5
PARAGRAPH_GAP = 7.0
TAG = 7.4

# A reference shows the head of its cited excerpt, so a printed report can be checked by eye
EXCERPT_CHARS = 220
SECTION_PATH_CHARS = 60
_SPEAKER_STAMP = re.compile(r"\[[^\[\]@]{1,40} @ \d{1,2}:\d{2}(?::\d{2})?\]:?")
_TABLE_DEBRIS = re.compile(r"(?:\|\s*)+\||\|?\s*:?-{3,}:?\s*\|?")


@dataclass(frozen=True)
class _Text:
    text: str
    family: str
    style: str
    size: float
    color: Tuple[int, int, int]


@dataclass(frozen=True)
class _Tag:
    number: int
    label: str


@dataclass(frozen=True)
class _Formula:
    svg: bytes
    width: float
    height: float
    ascent: float


class FpdfReportWriter(IReportDocumentWriter):
    def write(self, report: Report) -> bytes:
        return _Layout(report).build()


class _Pdf(FPDF):
    """fpdf2 with the running header (from page 2) and footer: the title, the date, the page count."""

    def __init__(self, title: str, date: str, page_of: str):
        super().__init__(orientation="P", unit="pt", format="A4")
        self.running_title = title
        self.date = date
        self.page_of = page_of

    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_font(fonts.SANS, "", 7.4)
        self.set_text_color(*FAINT)
        title = _fit(self, self.running_title, PAGE_W - 2 * MARGIN_X - 60)
        self.text(MARGIN_X, MARGIN_TOP - 30, title)
        self.text(PAGE_W - MARGIN_X - self.get_string_width("Onyx"), MARGIN_TOP - 30, "Onyx")
        self.set_draw_color(*RULE)
        self.set_line_width(0.5)
        self.line(MARGIN_X, MARGIN_TOP - 22, PAGE_W - MARGIN_X, MARGIN_TOP - 22)

    def stamp_footers(self) -> None:
        """
        The date and "Page n of N" on every page, once N is known. fpdf2's own page-count alias can't be used: it is
        substituted in the PDF's bytes, and with an embedded Unicode font the text is glyph ids, not characters.
        """
        total = self.page_no()
        for number in range(1, total + 1):
            self.page = number
            # fpdf2 skips a font or colour it believes is already selected, but that belief is the last page
            # touched, not this one: select something else first so both are written into this page
            self.set_font(fonts.SERIF, "", 1)
            self.set_text_color(0, 0, 0)
            self.set_font(fonts.SANS, "", 7.4)
            self.set_text_color(*FAINT)
            baseline = PAGE_H - MARGIN_BOTTOM + 32
            self.text(MARGIN_X, baseline, self.date)
            label = self.page_of.format(page=number, pages=total)
            self.text(PAGE_W - MARGIN_X - self.get_string_width(label), baseline, label)
        self.page = total


class _Layout:
    """One report, laid out top to bottom. `y` is the top of the next thing to be drawn."""

    def __init__(self, report: Report):
        self.report = report
        self.pack = get_language_pack(report.language)
        self.strings = self.pack.report
        stamp = report.completed_at or report.created_at
        self.pdf = _Pdf(report.display_title, stamp.strftime("%Y-%m-%d") if stamp else "", self.strings.page_of)
        fonts.register(self.pdf)
        self.pdf.set_auto_page_break(False)
        self.pdf.set_margins(MARGIN_X, MARGIN_TOP, MARGIN_X)
        self.left = MARGIN_X
        self.width = PAGE_W - 2 * MARGIN_X
        self.y = MARGIN_TOP
        # (background, bar colour, bar x) while a panel (summary, not-covered box, quote) is open
        self.panel: Optional[Tuple[Optional[Tuple[int, int, int]], Tuple[int, int, int], float]] = None
        self.entries: Dict[int, EvidenceEntry] = report.evidence_by_doc_number
        self.links: Dict[int, int] = {}

    # ==========================================
    # THE DOCUMENT
    # ==========================================

    def build(self) -> bytes:
        report, pdf = self.report, self.pdf
        pdf.set_title(report.display_title)
        pdf.set_subject(" ".join(report.topic.split()))
        pdf.set_author("Onyx")
        pdf.set_creator("Onyx")
        if report.language:
            pdf.set_lang(report.language)
        pdf.add_page()
        # A citation is drawn before the reference it points to; fpdf2 needs a destination for a link the moment it
        # is used, so each starts at page 1 and is moved to its reference row when that is drawn
        self.links = {n: pdf.add_link(page=1) for n in report.cited_doc_numbers}
        self._title_block()
        if report.executive_summary.strip():
            self._summary(report.executive_summary)
        for number, section in enumerate(report.written_sections, start=1):
            self._section_heading(f"{number}.", section.title)
            self._blocks(markdown_blocks.parse(section.markdown_body))
        if report.uncovered_sections:
            self._not_covered()
        self._references()
        pdf.stamp_footers()
        return bytes(pdf.output())

    def _title_block(self) -> None:
        pdf = self.pdf
        pdf.set_font(fonts.SANS, "B", 7.6)
        pdf.set_char_spacing(1.4)
        pdf.set_text_color(*ACCENT)
        pdf.text(self.left, self.y + 8, self.strings.document_label.upper())
        pdf.set_char_spacing(0)
        self.y += 22

        title = (TextRun(self.report.display_title, bold=True),)
        self._paragraph(title, size=24, leading=1.2, justify=False, after=10)

        lines = [line.removeprefix(">").strip() for line in provenance_lines(self.report)]
        if self.report.title and lines:
            self._paragraph((TextRun(lines.pop(0)),), family=fonts.SANS, size=9, color=MUTED, justify=False, after=3)
        if lines:
            self._paragraph((TextRun("  ·  ".join(lines)),), family=fonts.SANS, size=7.8, color=FAINT,
                            justify=False, leading=1.45, after=12)
        pdf.set_draw_color(*ACCENT)
        pdf.set_line_width(1.1)
        pdf.line(self.left, self.y, self.left + 46, self.y)
        self.y += 22

    def _summary(self, markdown: str) -> None:
        self._section_heading(None, self.strings.executive_summary, after=8)
        self._open_panel(PANEL, ACCENT)
        self._space(9)
        self._blocks(markdown_blocks.parse(markdown))
        self._close_panel(after=8)
        self._space(8)

    def _not_covered(self) -> None:
        self._section_heading(None, self.strings.not_covered_heading, after=8)
        items = "\n".join(self.strings.not_covered_item.format(title=s.title) for s in self.report.uncovered_sections)
        self._open_panel(NOTICE_PANEL, NOTICE_BAR)
        self._space(8)
        self._blocks(markdown_blocks.parse(items))
        self._close_panel(after=6)

    def _references(self) -> None:
        self._section_heading(None, self.strings.references_heading, after=10)
        cited = self.report.cited_doc_numbers
        if not cited:
            self._paragraph(markdown_blocks.parse_inline(self.strings.no_sources_cited), color=MUTED)
            return

        # Grouped under their source, in the order each source is first cited
        groups: Dict[str, List[int]] = {}
        names: Dict[str, str] = {}
        for number in cited:
            entry = self.entries.get(number)
            key = str(entry.source_id) if entry is not None else ""
            groups.setdefault(key, []).append(number)
            names[key] = entry.source_name if entry is not None else ""

        for key, numbers in groups.items():
            self._keep_together(46)
            if names[key]:
                self._paragraph((TextRun(names[key], bold=True),), family=fonts.SANS, size=8.8, justify=False,
                                after=5)
            for number in numbers:
                self._reference_row(number, self.entries.get(number))
            self._space(9)

    def _reference_row(self, number: int, entry: Optional[EvidenceEntry]) -> None:
        tag_width = 46.0
        if entry is None:
            where = markdown_blocks.parse_inline(self.strings.unresolved_reference)
            excerpt: Tuple[Inline, ...] = ()
        else:
            where = (TextRun(self._location(entry)),)
            # The excerpt's own words; a converted table's cell rules ("| --- |") are dropped, they read as noise
            # and a transcript's speaker stamps ("[speaker_0 @ 00:00:02]:") too, which the row already states
            digest = " ".join(_SPEAKER_STAMP.sub(" ", _TABLE_DEBRIS.sub(" ", entry.digest or "")).split())
            if len(digest) > EXCERPT_CHARS:
                digest = digest[:EXCERPT_CHARS].rsplit(" ", 1)[0] + "…"
            excerpt = (TextRun(f"“{digest}”", italic=True),) if digest else ()

        location_lines = self._lines(where, fonts.SANS, 8.3, MUTED, self.width - tag_width, justify=False)
        excerpt_lines = self._lines(excerpt, fonts.SERIF, 8.6, MUTED, self.width - tag_width, justify=False) \
            if excerpt else []
        heights = [self._line_height(line, 8.3, 1.4) for line in location_lines] + \
                  [self._line_height(line, 8.6, 1.42) for line in excerpt_lines]
        self._ensure(sum(heights) + 6)

        self.pdf.set_link(self.links[number], y=self.y - 4, page=self.pdf.page_no())
        tag_line = self._lines((Citation(number),), fonts.SANS, 8.3, ACCENT, tag_width, justify=False)[0]
        self._draw_line(tag_line, self.left, self._baseline(self.y, location_lines[0], 8.3, 1.4), link=False)

        saved = self.left
        self.left += tag_width
        for line in location_lines:
            self._place_line(line, 8.3, 1.4)
        for line in excerpt_lines:
            self._place_line(line, 8.6, 1.42)
        self.left = saved
        self._space(6)

    def _location(self, entry: EvidenceEntry) -> str:
        """Page, lines or timespan, then only the innermost heading: a document's full path is mostly noise."""
        path = entry.section_path
        if path:
            innermost = path.split(self.pack.documents.section_separator.strip())[-1].strip()
            path = innermost if len(innermost) <= SECTION_PATH_CHARS else innermost[:SECTION_PATH_CHARS].rstrip() + "…"
        where = format_location(entry.anchor_kind, entry.start_sec, entry.end_sec, entry.page_start,
                                entry.page_end, path, self.pack.documents)
        if entry.anchor_kind is AnchorKind.TIME and entry.speaker_labels:
            where = f"{where}  ·  {', '.join(entry.speaker_labels)}"
        return where

    # ==========================================
    # BLOCKS
    # ==========================================

    def _section_heading(self, number: Optional[str], title: str, after: float = 6) -> None:
        size = 15.0
        inlines: List[Inline] = [TextRun(title, bold=True)]
        lines = self._lines(tuple(inlines), fonts.SERIF, size, TEXT, self.width - (24 if number else 0), justify=False)
        height = sum(self._line_height(line, size, 1.25) for line in lines)
        if self.y > MARGIN_TOP + 1:
            self._space(16)
        # A heading never ends a page: room for it and about three lines of what follows, or a new page
        self._keep_together(height + 3 * BODY * LEADING)
        self.pdf.set_y(self.y)
        self.pdf.start_section(title, level=0)
        if number:
            self.pdf.set_font(fonts.SERIF, "B", size)
            self.pdf.set_text_color(*ACCENT)
            self.pdf.text(self.left, self._baseline(self.y, lines[0], size, 1.25), number)
        saved = self.left
        self.left += 24 if number else 0
        for line in lines:
            self._place_line(line, size, 1.25)
        self.left = saved
        self._space(after)

    def _blocks(self, blocks: Sequence[Block], color: Tuple[int, int, int] = TEXT, italic: bool = False) -> None:
        for block in blocks:
            if isinstance(block, Paragraph):
                inlines = tuple(TextRun(i.text, i.bold, True, i.code) if italic and isinstance(i, TextRun) else i
                                for i in block.inlines)
                self._paragraph(inlines, color=color)
            elif isinstance(block, Heading):
                size = 12.5 if block.level <= 3 else BODY
                self._space(5)
                self._keep_together(size * 1.3 + 2 * BODY * LEADING)
                bold = tuple(TextRun(i.text, True, i.italic, i.code) if isinstance(i, TextRun) else i
                             for i in block.inlines)
                self._paragraph(bold, size=size, leading=1.3, justify=False, after=3)
            elif isinstance(block, ListBlock):
                self._list(block, color)
            elif isinstance(block, MathBlock):
                self._display_math(block.latex)
            elif isinstance(block, TableBlock):
                self._table(block)
            elif isinstance(block, Quote):
                saved_left, saved_width, saved_panel = self.left, self.width, self.panel
                self.panel = (None, RULE, self.left + 2)
                self.left += 14
                self.width -= 14
                self._blocks(block.blocks, color=MUTED, italic=True)
                self.left, self.width, self.panel = saved_left, saved_width, saved_panel
            elif isinstance(block, Rule):
                self._space(4)
                self.pdf.set_draw_color(*RULE)
                self.pdf.set_line_width(0.6)
                self.pdf.line(self.left, self.y, self.left + self.width, self.y)
                self._space(10)

    def _paragraph(self, inlines: Tuple[Inline, ...], family: str = fonts.SERIF, size: float = BODY,
                   color: Tuple[int, int, int] = TEXT, justify: bool = True, leading: float = LEADING,
                   after: float = PARAGRAPH_GAP, marker: Optional[str] = None) -> None:
        lines = self._lines(inlines, family, size, color, self.width, justify)
        for index, line in enumerate(lines):
            top = self._place_line(line, size, leading)
            if index == 0 and marker:
                self.pdf.set_font(fonts.SERIF, "", size)
                self.pdf.set_text_color(*ACCENT)
                self.pdf.text(self.left - 13, self._baseline(top, line, size, leading), marker)
        self._space(after)

    def _list(self, block: ListBlock, color: Tuple[int, int, int]) -> None:
        indent = 18.0
        self.left += indent
        self.width -= indent
        for index, item in enumerate(block.items):
            marker = f"{block.start + index}." if block.ordered else "•"
            for position, inner in enumerate(item):
                if position == 0 and isinstance(inner, Paragraph):
                    self._paragraph(inner.inlines, color=color, after=3, marker=marker)
                else:
                    self._blocks([inner], color=color)
        self.left -= indent
        self.width += indent
        self._space(PARAGRAPH_GAP - 3)

    def _display_math(self, latex: str) -> None:
        try:
            formula = math_render.render(latex, BODY * 1.12, display=True)
        except MathRenderError:
            self._paragraph((TextRun(f"$${latex}$$", code=True),), justify=False)
            return
        self._space(3)
        self._ensure(formula.height + 6)
        self._slice(formula.height + 6)
        x = self.left + max(0.0, (self.width - formula.width) / 2)
        self.pdf.image(BytesIO(formula.svg), x=x, y=self.y + 3, w=formula.width, h=formula.height)
        self.y += formula.height + 6
        self._space(PARAGRAPH_GAP)

    def _table(self, table: TableBlock) -> None:
        columns = max([len(table.header)] + [len(row) for row in table.rows] or [0])
        if columns == 0:
            return
        size, pad = 8.6, 5.0
        header = list(table.header) + [()] * (columns - len(table.header))
        rows = [list(row) + [()] * (columns - len(row)) for row in table.rows]

        # Columns share the width in proportion to their content, within reason
        natural = [0.0] * columns
        for row in [header] + rows:
            for c, cell in enumerate(row):
                width = sum(a.width for a in self._atoms(cell, fonts.SERIF, size, TEXT) if not a.newline)
                natural[c] = max(natural[c], min(width, self.width * 0.6) + 2 * pad)
        total = sum(natural) or 1.0
        widths = [self.width * n / total for n in natural]

        def lay(row, bold):
            cells = [self._lines(tuple(TextRun(i.text, bold or i.bold, i.italic, i.code) if isinstance(i, TextRun)
                                       else i for i in cell), fonts.SERIF, size, TEXT, w - 2 * pad, justify=False)
                     for cell, w in zip(row, widths)]
            height = max(sum(self._line_height(l, size, 1.35) for l in lines) for lines in cells) + 2 * pad
            return cells, height

        head_cells, head_height = lay(header, True)

        def draw(cells, height, shaded):
            if shaded:
                self.pdf.set_fill_color(*PANEL)
                self.pdf.rect(self.left, self.y, self.width, height, style="F")
            x = self.left
            for lines, w in zip(cells, widths):
                top = self.y + pad
                for line in lines:
                    self._draw_line(line, x + pad, self._baseline(top, line, size, 1.35))
                    top += self._line_height(line, size, 1.35)
                x += w
            self.y += height
            self.pdf.set_draw_color(*RULE)
            self.pdf.set_line_width(0.6)
            self.pdf.line(self.left, self.y, self.left + self.width, self.y)

        self._space(2)
        self._ensure(head_height + 24)
        if any(header):
            draw(head_cells, head_height, True)
        for row in rows:
            cells, height = lay(row, False)
            if self.y + height > PAGE_H - MARGIN_BOTTOM:
                self._new_page()
                if any(header):
                    draw(head_cells, head_height, True)  # the header is repeated on every page the table spans
            draw(cells, height, False)
        self._space(PARAGRAPH_GAP + 3)

    # ==========================================
    # LINES
    # ==========================================

    def _atoms(self, inlines: Iterable[Inline], family: str, size: float, color: Tuple[int, int, int]) -> List[Atom]:
        atoms: List[Atom] = []
        for item in inlines:
            if isinstance(item, TextRun):
                run_family = fonts.SANS if item.code else family
                run_size = size * 0.92 if item.code else size
                style = ("B" if item.bold else "") + ("I" if item.italic and run_family == fonts.SERIF else "")
                self.pdf.set_font(run_family, style, run_size)
                space = self.pdf.get_string_width(" ")
                for piece in re.split(r"(\s+)", item.text):
                    if not piece:
                        continue
                    if piece.isspace():
                        atoms.append(Atom(width=space, glue=True))
                    else:
                        atoms.append(Atom(width=self.pdf.get_string_width(piece), ascent=run_size * 0.74,
                                          descent=run_size * 0.24,
                                          payload=_Text(piece, run_family, style, run_size,
                                                        MUTED if item.code else color)))
            elif isinstance(item, Citation):
                label = f"Doc {item.number}"
                self.pdf.set_font(fonts.SANS, "", TAG)
                width = self.pdf.get_string_width(label) + 6.0
                atoms.append(Atom(width=width + 3.0, ascent=TAG * 1.0, descent=TAG * 0.3,
                                  payload=_Tag(item.number, label)))
            elif isinstance(item, InlineMath):
                try:
                    formula = math_render.render(item.latex, size * 1.06, display=False)
                    atoms.append(Atom(width=formula.width, ascent=formula.ascent, descent=formula.descent,
                                      payload=_Formula(formula.svg, formula.width, formula.height, formula.ascent)))
                except MathRenderError:
                    # Never fail the export over one formula: show what the writer wrote
                    self.pdf.set_font(fonts.SANS, "", size * 0.9)
                    text = f"${item.latex}$"
                    atoms.append(Atom(width=self.pdf.get_string_width(text), ascent=size * 0.74, descent=size * 0.24,
                                      payload=_Text(text, fonts.SANS, "", size * 0.9, MUTED)))
            elif isinstance(item, LineBreak):
                atoms.append(Atom(width=0, newline=True))
        return atoms

    def _lines(self, inlines: Tuple[Inline, ...], family: str, size: float, color: Tuple[int, int, int],
               width: float, justify: bool) -> List[Line]:
        return break_lines(self._atoms(inlines, family, size, color), width, justify)

    @staticmethod
    def _line_height(line: Line, size: float, leading: float) -> float:
        ascent = max(line.ascent, size * 0.74)
        descent = max(line.descent, size * 0.24)
        return max(size * leading, ascent + descent + size * 0.4)

    @staticmethod
    def _baseline(top: float, line: Line, size: float, leading: float) -> float:
        ascent = max(line.ascent, size * 0.74)
        descent = max(line.descent, size * 0.24)
        return top + (_Layout._line_height(line, size, leading) - (ascent + descent)) / 2 + ascent

    def _place_line(self, line: Line, size: float, leading: float) -> float:
        """Draws one line at the cursor, turning the page first if it would not fit. Returns the line's top."""
        height = self._line_height(line, size, leading)
        self._ensure(height)
        self._slice(height)
        top = self.y
        self._draw_line(line, self.left, self._baseline(top, line, size, leading))
        self.y += height
        return top

    def _draw_line(self, line: Line, x0: float, baseline: float, link: bool = True) -> None:
        pdf = self.pdf
        for x, atom in line.placed:
            payload = atom.payload
            if isinstance(payload, _Text):
                pdf.set_font(payload.family, payload.style, payload.size)
                pdf.set_text_color(*payload.color)
                pdf.text(x0 + x, baseline, payload.text)
            elif isinstance(payload, _Formula):
                pdf.image(BytesIO(payload.svg), x=x0 + x, y=baseline - payload.ascent, w=payload.width,
                          h=payload.height)
            elif isinstance(payload, _Tag):
                self._draw_tag(payload, x0 + x + 1.5, baseline, atom.width - 3.0, link)

    def _draw_tag(self, tag: _Tag, x: float, baseline: float, width: float, link: bool) -> None:
        pdf = self.pdf
        top, height = baseline - TAG * 0.98, TAG * 1.32
        pdf.set_draw_color(*ACCENT)
        pdf.set_line_width(0.55)
        pdf.rect(x, top, width, height, style="D", round_corners=True, corner_radius=1.6)
        pdf.set_font(fonts.SANS, "", TAG)
        pdf.set_text_color(*ACCENT)
        pdf.text(x + 3.0, baseline - 0.4, tag.label)
        if link and tag.number in self.links:
            pdf.link(x, top, width, height, self.links[tag.number])

    # ==========================================
    # THE CURSOR
    # ==========================================

    def _space(self, height: float) -> None:
        if self.y + height > PAGE_H - MARGIN_BOTTOM:
            self._new_page()
            return
        self._slice(height)
        self.y += height

    def _ensure(self, height: float) -> None:
        if self.y + height > PAGE_H - MARGIN_BOTTOM and self.y > MARGIN_TOP + 1:
            self._new_page()

    def _keep_together(self, height: float) -> None:
        self._ensure(min(height, PAGE_H - MARGIN_TOP - MARGIN_BOTTOM))

    def _new_page(self) -> None:
        self.pdf.add_page()
        self.y = MARGIN_TOP

    def _open_panel(self, background: Tuple[int, int, int], bar: Tuple[int, int, int]) -> None:
        self.panel = (background, bar, self.left - 12)
        self.left += 4
        self.width -= 8

    def _close_panel(self, after: float) -> None:
        self._space(after)
        self.panel = None
        self.left -= 4
        self.width += 8

    def _slice(self, height: float) -> None:
        """Paints the open panel's background and bar behind the next `height` points."""
        if self.panel is None or height <= 0:
            return
        background, bar, bar_x = self.panel
        if background is not None:
            self.pdf.set_fill_color(*background)
            self.pdf.rect(bar_x, self.y, PAGE_W - MARGIN_X + 12 - bar_x, height, style="F")
        self.pdf.set_fill_color(*bar)
        self.pdf.rect(bar_x, self.y, 2.2, height, style="F")


def _fit(pdf: FPDF, text: str, width: float) -> str:
    """`text`, cut with an ellipsis to fit `width` in the current font."""
    if pdf.get_string_width(text) <= width:
        return text
    while text and pdf.get_string_width(text + "…") > width:
        text = text[:-1]
    return text.rstrip() + "…"
