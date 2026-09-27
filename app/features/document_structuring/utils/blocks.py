# Path: app/features/document_structuring/utils/blocks.py

"""
Pages of text -> an ordered list of blocks (headings, paragraphs, tables), without the page furniture.
"""

import re
from collections import Counter
from typing import Callable, List, Optional, Pattern, Sequence, Tuple

from app.features.document_structuring.domain.entities import Block
from app.features.document_structuring.domain.enums import BlockKind
from app.features.document_structuring.utils.headings import heading_level

# Only the top and bottom lines of a page are checked for repeated headers/footers, so a phrase repeated in the
# body ("The Tenant shall...") is never mistaken for page furniture
_FURNITURE_LINES = 2
_MAX_FURNITURE_WORDS = 12
_PAGE_NUMBER = re.compile(
    r"^(?:[-–—]\s*)?(?:page|stran|strana|str\.?|p\.?)?\s*(\d+)\s*(?:(?:of|od|/)\s*\d+)?(?:\s*[-–—])?$",
    re.IGNORECASE)
# A printed page number may lag the physical one (an unnumbered cover page, a front matter section)
_PAGE_NUMBER_DRIFT = 2
_DIGITS = re.compile(r"\d+")


def parse_blocks(
        pages: Sequence[Tuple[Optional[int], str]],
        clause_patterns: List[Pattern],
        normalize: Callable[[str], str] = lambda text: text
) -> List[Block]:
    """
    `pages` are (page number, text) pairs in order. Paragraphs are separated by blank lines; a heading line
    inside a paragraph splits it (OCR output rarely has blank lines between a heading and its body). Hard-wrapped
    lines of a paragraph are joined with a space -- whitespace only, so the text stays verbatim.
    """
    furniture = _repeated_furniture(pages)
    blocks: List[Block] = []
    for page_no, text in pages:
        blocks.extend(_page_blocks(page_no, _without_furniture(text, page_no, furniture), clause_patterns, normalize))
    return blocks


def _without_furniture(text: str, page_no: Optional[int], furniture: set) -> List[Tuple[int, str]]:
    """
    The page's (line number, line) pairs minus its running header, footer and page number. Only the top and
    bottom lines are candidates: "2025" in a table is a year, and a sentence that repeats across pages is still
    content. Dropped lines keep their number, so the remaining lines are numbered as in the source.
    """
    lines = text.split("\n")
    content = [i for i, line in enumerate(lines) if line.strip()]
    edges = set(content[:_FURNITURE_LINES] + content[-_FURNITURE_LINES:])
    return [(i + 1, line) for i, line in enumerate(lines)
            if not (i in edges and _is_furniture(line, page_no, furniture))]


def _page_blocks(page_no, lines, clause_patterns, normalize) -> List[Block]:
    blocks: List[Block] = []
    paragraph: List[Tuple[int, str]] = []
    table: List[Tuple[int, str]] = []

    def block(kind, numbered, text, level=0) -> Block:
        return Block(kind=kind, text=normalize(text), page_no=page_no, level=level,
                     line_start=numbered[0][0], line_end=numbered[-1][0])

    def flush_paragraph():
        if paragraph:
            blocks.append(block(BlockKind.PARAGRAPH, paragraph, " ".join(line for _, line in paragraph)))
            paragraph.clear()

    def flush_table():
        if table:
            blocks.append(block(BlockKind.TABLE, table, "\n".join(line for _, line in table)))
            table.clear()

    for line_no, raw in lines:
        line = raw.strip()
        if not line:
            flush_paragraph()
            flush_table()
            continue
        if line.startswith("|"):
            flush_paragraph()
            table.append((line_no, line))
            continue
        flush_table()
        heading = heading_level(line, clause_patterns)
        if heading:
            flush_paragraph()
            level, heading_text = heading
            blocks.append(block(BlockKind.HEADING, [(line_no, line)], heading_text, level))
            continue
        paragraph.append((line_no, line))

    flush_paragraph()
    flush_table()
    return blocks


def _repeated_furniture(pages: Sequence[Tuple[Optional[int], str]]) -> set:
    """Lines at the top or bottom of at least half the pages (and at least 3): running headers and footers."""
    if len(pages) < 3:
        return set()
    counts: Counter = Counter()
    for _, text in pages:
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        edge = set(lines[:_FURNITURE_LINES] + lines[-_FURNITURE_LINES:])
        counts.update(_furniture_key(line) for line in edge if _could_be_furniture(line))
    threshold = max(3, (len(pages) + 1) // 2)
    return {key for key, count in counts.items() if count >= threshold}


def _could_be_furniture(line: str) -> bool:
    """
    Running headers and footers are short labels ("ACME LTD - CONFIDENTIAL", "Page 3 of 12"), never sentences.
    On a short page the body sits in the edge lines too, and a sentence repeated there is still content.
    """
    return len(line.split()) <= _MAX_FURNITURE_WORDS and not line.rstrip().endswith((".", ";", ":", ","))


def _furniture_key(line: str) -> str:
    # "Page 3 of 12" and "Page 4 of 12" are the same footer
    return _DIGITS.sub("#", line.strip().lower())


def _is_furniture(line: str, page_no: Optional[int], furniture: set) -> bool:
    stripped = line.strip()
    if _furniture_key(stripped) in furniture:
        return True
    number = _PAGE_NUMBER.match(stripped)
    return bool(number and page_no is not None and abs(int(number.group(1)) - page_no) <= _PAGE_NUMBER_DRIFT)
