# Path: app/features/report_export/domain/entities.py

"""
What an exported report is laid out from: a section body's Markdown, parsed into a few kinds of block and inline.

Small on purpose. A report is headings, prose, lists, an occasional table and formula, and `[Doc N]` citations;
anything Markdown can say beyond that (images, HTML, footnotes) is flattened to text rather than modelled.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union


# ---------- inlines ----------

@dataclass(frozen=True)
class TextRun:
    text: str
    bold: bool = False
    italic: bool = False
    code: bool = False


@dataclass(frozen=True)
class Citation:
    """`[Doc N]`: drawn as a tag that links to the reference it resolves to."""
    number: int


@dataclass(frozen=True)
class InlineMath:
    latex: str


@dataclass(frozen=True)
class LineBreak:
    pass


Inline = Union[TextRun, Citation, InlineMath, LineBreak]


# ---------- blocks ----------

@dataclass(frozen=True)
class Paragraph:
    inlines: Tuple[Inline, ...]


@dataclass(frozen=True)
class Heading:
    """A heading inside a section's body (the section titles themselves come from the report, not its Markdown)."""
    level: int
    inlines: Tuple[Inline, ...]


@dataclass(frozen=True)
class MathBlock:
    """A formula on a line of its own (`$$…$$` or `\\[…\\]` as a whole paragraph)."""
    latex: str


@dataclass(frozen=True)
class ListBlock:
    ordered: bool
    items: Tuple[Tuple["Block", ...], ...]
    start: int = 1


@dataclass(frozen=True)
class TableBlock:
    header: Tuple[Tuple[Inline, ...], ...]
    rows: Tuple[Tuple[Tuple[Inline, ...], ...], ...]


@dataclass(frozen=True)
class Quote:
    blocks: Tuple["Block", ...]


@dataclass(frozen=True)
class Rule:
    pass


Block = Union[Paragraph, Heading, MathBlock, ListBlock, TableBlock, Quote, Rule]


@dataclass(frozen=True)
class MathSpan:
    """A formula found in Markdown text: its LaTeX, and whether it was written to stand on its own line."""
    latex: str
    display: bool = False


@dataclass(frozen=True)
class RenderedMath:
    """A typeset formula: SVG, its size in points, and how far it rises above the text baseline."""
    svg: bytes
    width: float
    height: float
    ascent: float

    @property
    def descent(self) -> float:
        return self.height - self.ascent
