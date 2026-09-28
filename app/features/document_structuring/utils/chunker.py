# Path: app/features/document_structuring/utils/chunker.py

"""
Blocks -> chunks along the document's own sections.

A chunk never crosses a clause or title heading, never splits a table row, and never overlaps its neighbour
(clause boundaries make overlap unnecessary, and without it no sentence is shown -- or cited -- twice). A short
one-page document is one chunk: a receipt split in half answers nothing. A short document spanning several pages
is still split by section, or every citation of it would name all of its pages.
"""

import re
from typing import List, Optional, Tuple

from app.features.document_structuring.domain.entities import Block, StructuredChunk
from app.features.document_structuring.domain.enums import BlockKind
from app.features.document_structuring.utils.headings import CLAUSE_LEVEL

_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+")


def build_chunks(blocks: List[Block], max_words: int, short_document_words: int, separator: str) -> List[StructuredChunk]:
    if not blocks:
        return []

    pieces = _sized_pieces(blocks, max_words)
    pages = {block.page_no for block in blocks}
    total_words = sum(_words(block.text) for block in blocks)
    # "Short" only when it is also one page (or unpaged): then splitting it gains no citation precision
    if len(pages) > 1:
        short_document_words = 0

    groups: List[List[Tuple[Block, Optional[str]]]] = []
    current: List[Tuple[Block, Optional[str]]] = []
    current_words = 0
    path: List[Tuple[int, str]] = []

    def has_body() -> bool:
        return any(block.kind is not BlockKind.HEADING for block, _ in current)

    for block in pieces:
        if block.kind is BlockKind.HEADING:
            while path and path[-1][0] >= block.level:
                path.pop()
            path.append((block.level, block.text))
            # A title or clause heading starts a new chunk; a sub-clause heading only does when the chunk is full
            if has_body() and total_words >= short_document_words and block.level <= CLAUSE_LEVEL:
                groups.append(current)
                current, current_words = [], 0
        words = _words(block.text)
        if has_body() and total_words >= short_document_words and current_words + words > max_words:
            groups.append(current)
            current, current_words = [], 0
        current.append((block, separator.join(text for _, text in path) or None))
        current_words += words

    if current:
        if groups and not has_body():
            # Trailing headings with no body of their own belong to the chunk before them
            groups[-1].extend(current)
        else:
            groups.append(current)

    return [_chunk(ordinal, group) for ordinal, group in enumerate(groups)]


def _chunk(ordinal: int, group: List[Tuple[Block, Optional[str]]]) -> StructuredChunk:
    pages = [block.page_no for block, _ in group if block.page_no is not None]
    starts = [block.line_start for block, _ in group if block.line_start is not None]
    ends = [block.line_end for block, _ in group if block.line_end is not None]
    # The chunk's section is where its body sits: the path at its first non-heading block
    body_paths = [path for block, path in group if block.kind is not BlockKind.HEADING]
    section_path = body_paths[0] if body_paths else group[-1][1]
    return StructuredChunk(
        ordinal=ordinal,
        text="\n\n".join(block.text for block, _ in group),
        page_start=min(pages) if pages else None,
        page_end=max(pages) if pages else None,
        line_start=min(starts) if starts else None,
        line_end=max(ends) if ends else None,
        section_path=section_path,
    )


def _sized_pieces(blocks: List[Block], max_words: int) -> List[Block]:
    """Blocks no longer than `max_words`: long paragraphs split at sentence ends, long tables into row groups."""
    pieces: List[Block] = []
    for block in blocks:
        if _words(block.text) <= max_words or block.kind is BlockKind.HEADING:
            pieces.append(block)
        elif block.kind is BlockKind.TABLE:
            pieces.extend(_split_table(block, max_words))
        else:
            pieces.extend(_split_paragraph(block, max_words))
    return pieces


def _split_paragraph(block: Block, max_words: int) -> List[Block]:
    parts, current = [], []
    for sentence in _SENTENCE_END.split(block.text):
        if current and _words(" ".join(current + [sentence])) > max_words:
            parts.append(" ".join(current))
            current = []
        current.append(sentence)
    if current:
        parts.append(" ".join(current))
    # Each part keeps the whole paragraph's lines: a sentence split mid-line has no line of its own, and a range
    # that covers the part is honest where a narrower guess might not be
    return [block.model_copy(update={"text": part}) for part in parts]


def _split_table(block: Block, max_words: int) -> List[Block]:
    """Row groups under the budget, each repeating the header row (and Markdown's separator row) on top."""
    rows = block.text.split("\n")
    header = rows[:2] if len(rows) > 1 and set(rows[1].replace("|", "").strip()) <= set("-: ") else rows[:1]
    body = rows[len(header):]
    header_words = _words("\n".join(header))

    groups, current = [], []
    for row in body:
        if current and header_words + _words("\n".join(current + [row])) > max_words:
            groups.append(current)
            current = []
        current.append(row)
    if current:
        groups.append(current)
    return [block.model_copy(update={"text": "\n".join(header + group)}) for group in groups]


def _words(text: str) -> int:
    return len(text.split())
