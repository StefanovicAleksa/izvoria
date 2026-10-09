# Path: app/features/report_export/utils/markdown_blocks.py

"""
A section body's Markdown, as the blocks and inlines the PDF is laid out from.

markdown-it-py reads it (CommonMark plus GFM tables, the dialect the writer produces). Formulas are swapped for
placeholders first (`math_text.protect`): inside a formula, `_` and `\\` would otherwise be read as emphasis and
escapes, and `$E_p$ … $Q_S$` would come back as italics. `[Doc N]` stays plain text through the parser (there are
no link definitions) and is turned into a citation here.
"""

import re
from typing import List, Optional, Sequence, Tuple

from markdown_it import MarkdownIt
from markdown_it.token import Token

from app.features.report_export.domain.entities import (
    Block, Citation, Heading, Inline, InlineMath, LineBreak, ListBlock, MathBlock, MathSpan, Paragraph, Quote, Rule,
    TableBlock, TextRun,
)
from app.features.report_export.utils.math_text import PLACEHOLDER, protect

_PARSER = MarkdownIt("commonmark").enable("table")
_CITATION = re.compile(r"\[Doc (\d+)\]")
_TOKEN = re.compile(r"\[Doc (?P<citation>\d+)\]|" + PLACEHOLDER.pattern.replace("(\\d+)", "(?P<math>\\d+)"))


def parse(markdown: str) -> List[Block]:
    protected, spans = protect(markdown)
    tokens = _PARSER.parse(protected)
    blocks, _ = _blocks(tokens, 0, len(tokens), spans)
    return blocks


def parse_inline(text: str) -> Tuple[Inline, ...]:
    """One line of Markdown (a heading, a list item from a language pack) as inlines."""
    blocks = parse(text)
    if blocks and isinstance(blocks[0], Paragraph):
        return blocks[0].inlines
    if blocks and isinstance(blocks[0], ListBlock) and blocks[0].items and blocks[0].items[0]:
        first = blocks[0].items[0][0]
        if isinstance(first, Paragraph):
            return first.inlines
    return (TextRun(text),)


# ---------- blocks ----------

def _blocks(tokens: Sequence[Token], start: int, stop: int, spans: List[MathSpan]) -> Tuple[List[Block], int]:
    blocks: List[Block] = []
    i = start
    while i < stop:
        token = tokens[i]
        kind = token.type
        if kind == "paragraph_open":
            inline = tokens[i + 1]
            blocks.append(_paragraph(inline, spans))
            i += 3
        elif kind == "heading_open":
            level = int(token.tag[1:])
            blocks.append(Heading(level=level, inlines=_inlines(tokens[i + 1], spans)))
            i += 3
        elif kind in ("bullet_list_open", "ordered_list_open"):
            close = _closing(tokens, i)
            blocks.append(_list(tokens, i, close, spans))
            i = close + 1
        elif kind == "blockquote_open":
            close = _closing(tokens, i)
            inner, _ = _blocks(tokens, i + 1, close, spans)
            blocks.append(Quote(blocks=tuple(inner)))
            i = close + 1
        elif kind == "table_open":
            close = _closing(tokens, i)
            blocks.append(_table(tokens, i, close, spans))
            i = close + 1
        elif kind in ("fence", "code_block"):
            text = _restore(token.content.rstrip("\n"), spans)
            blocks.append(Paragraph(inlines=tuple(
                item for n, line in enumerate(text.split("\n"))
                for item in ((LineBreak(),) if n else ()) + (TextRun(line, code=True),)
            )))
            i += 1
        elif kind == "hr":
            blocks.append(Rule())
            i += 1
        elif kind == "html_block":
            blocks.append(Paragraph(inlines=(TextRun(_restore(token.content.strip(), spans)),)))
            i += 1
        else:
            i += 1
    return blocks, i


def _paragraph(inline: Token, spans: List[MathSpan]) -> Block:
    # A paragraph that is nothing but one display formula is set on its own line, centred
    match = PLACEHOLDER.fullmatch(inline.content.strip())
    if match is not None:
        span = spans[int(match.group(1))]
        if span.display:
            return MathBlock(latex=span.latex)
    return Paragraph(inlines=_inlines(inline, spans))


def _list(tokens: Sequence[Token], start: int, stop: int, spans: List[MathSpan]) -> ListBlock:
    opening = tokens[start]
    items: List[Tuple[Block, ...]] = []
    i = start + 1
    while i < stop:
        if tokens[i].type == "list_item_open":
            close = _closing(tokens, i)
            inner, _ = _blocks(tokens, i + 1, close, spans)
            items.append(tuple(inner))
            i = close + 1
        else:
            i += 1
    ordered = opening.type == "ordered_list_open"
    first = int(opening.attrGet("start") or 1) if ordered else 1
    return ListBlock(ordered=ordered, items=tuple(items), start=first)


def _table(tokens: Sequence[Token], start: int, stop: int, spans: List[MathSpan]) -> TableBlock:
    header: List[Tuple[Inline, ...]] = []
    rows: List[Tuple[Tuple[Inline, ...], ...]] = []
    row: Optional[List[Tuple[Inline, ...]]] = None
    in_head = False
    for token in tokens[start:stop]:
        if token.type == "thead_open":
            in_head = True
        elif token.type == "thead_close":
            in_head = False
        elif token.type == "tr_open":
            row = []
        elif token.type == "tr_close" and row is not None:
            if in_head:
                header = row
            else:
                rows.append(tuple(row))
            row = None
        elif token.type == "inline" and row is not None:
            row.append(_inlines(token, spans))
    return TableBlock(header=tuple(header), rows=tuple(rows))


def _closing(tokens: Sequence[Token], start: int) -> int:
    """The index of the token that closes the one at `start` (same nesting level)."""
    depth = 0
    for i in range(start, len(tokens)):
        depth += tokens[i].nesting
        if depth == 0:
            return i
    return len(tokens) - 1


# ---------- inlines ----------

def _inlines(inline: Token, spans: List[MathSpan]) -> Tuple[Inline, ...]:
    result: List[Inline] = []
    bold = italic = 0
    for child in inline.children or []:
        kind = child.type
        if kind == "text":
            result.extend(_split(child.content, spans, bold > 0, italic > 0))
        elif kind == "code_inline":
            result.append(TextRun(_restore(child.content, spans), code=True))
        elif kind == "softbreak":
            result.append(TextRun(" ", bold > 0, italic > 0))
        elif kind == "hardbreak":
            result.append(LineBreak())
        elif kind == "strong_open":
            bold += 1
        elif kind == "strong_close":
            bold = max(0, bold - 1)
        elif kind == "em_open":
            italic += 1
        elif kind == "em_close":
            italic = max(0, italic - 1)
        elif kind in ("html_inline", "image"):
            result.extend(_split(child.content, spans, bold > 0, italic > 0))
        # link_open/close: the link's text is kept, its target dropped (a report links nowhere but its references)
    return tuple(_merge(result))


def _split(text: str, spans: List[MathSpan], bold: bool, italic: bool) -> List[Inline]:
    parts: List[Inline] = []
    position = 0
    for match in _TOKEN.finditer(text):
        if match.start() > position:
            parts.append(TextRun(text[position:match.start()], bold, italic))
        if match.group("citation") is not None:
            parts.append(Citation(number=int(match.group("citation"))))
        else:
            parts.append(InlineMath(latex=spans[int(match.group("math"))].latex))
        position = match.end()
    if position < len(text):
        parts.append(TextRun(text[position:], bold, italic))
    return parts


def _restore(text: str, spans: List[MathSpan]) -> str:
    """Puts a formula back as LaTeX where it is shown as text (code)."""
    return PLACEHOLDER.sub(lambda m: f"${spans[int(m.group(1))].latex}$", text)


def _merge(items: List[Inline]) -> List[Inline]:
    """Adjacent runs of the same style become one, so words are measured whole."""
    merged: List[Inline] = []
    for item in items:
        previous = merged[-1] if merged else None
        if (isinstance(item, TextRun) and isinstance(previous, TextRun)
                and (previous.bold, previous.italic, previous.code) == (item.bold, item.italic, item.code)):
            merged[-1] = TextRun(previous.text + item.text, item.bold, item.italic, item.code)
        else:
            merged.append(item)
    return merged
