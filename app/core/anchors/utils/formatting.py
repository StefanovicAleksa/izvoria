# Path: app/core/anchors/utils/formatting.py

"""
One place that turns a chunk's anchor into the text a citation shows: "00:12:31–00:14:02", "p. 7", "pp. 7–8",
"lines 120–164", a section path, or "image". Every citation header, reference table and evidence card goes through `format_anchor`, so a
document chunk never shows up as "00:00:00".
"""

from typing import Optional

from app.core.anchors.domain.enums import AnchorKind
from app.core.language.domain.entities import DocumentRules
from app.core.language.service.language_api import get_language_pack


def format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def format_pages(page_start: Optional[int], page_end: Optional[int], rules: Optional[DocumentRules] = None) -> str:
    rules = rules or get_language_pack().documents
    if page_start is None:
        return ""
    if page_end is None or page_end == page_start:
        return f"{rules.page_label} {page_start}"
    return f"{rules.pages_label} {page_start}–{page_end}"


def format_lines(line_start: Optional[int], line_end: Optional[int], rules: Optional[DocumentRules] = None) -> str:
    rules = rules or get_language_pack().documents
    if line_start is None:
        return ""
    if line_end is None or line_end == line_start:
        return f"{rules.line_label} {line_start}"
    return f"{rules.lines_label} {line_start}–{line_end}"


def format_anchor(
        anchor_kind: AnchorKind,
        start_sec: float = 0.0,
        end_sec: float = 0.0,
        page_start: Optional[int] = None,
        page_end: Optional[int] = None,
        section_path: Optional[str] = None,
        rules: Optional[DocumentRules] = None
) -> str:
    """
    The location part of a citation. A page or line anchor falls back to its section path when its number is
    unknown, and a section anchor without a path renders as an empty string rather than inventing a location.
    A LINE anchor's lines arrive in `page_start`/`page_end`.
    """
    kind = AnchorKind(anchor_kind)
    if kind is AnchorKind.TIME:
        return f"{format_timestamp(start_sec)}–{format_timestamp(end_sec)}"
    if kind is AnchorKind.PAGE and page_start is not None:
        return format_pages(page_start, page_end, rules)
    if kind is AnchorKind.LINE and page_start is not None:
        return format_lines(page_start, page_end, rules)
    if kind is AnchorKind.IMAGE:
        return (rules or get_language_pack().documents).image_label
    return section_path or ""


def format_location(
        anchor_kind: AnchorKind,
        start_sec: float = 0.0,
        end_sec: float = 0.0,
        page_start: Optional[int] = None,
        page_end: Optional[int] = None,
        section_path: Optional[str] = None,
        rules: Optional[DocumentRules] = None
) -> str:
    """
    The anchor plus, for a paged or line-numbered document, the section it sits in: "p. 7 · Article 7 Termination",
    "lines 120–164 · Chapter 4".
    """
    anchor = format_anchor(anchor_kind, start_sec, end_sec, page_start, page_end, section_path, rules)
    if (AnchorKind(anchor_kind) in (AnchorKind.PAGE, AnchorKind.LINE)
            and section_path and page_start is not None):
        return f"{anchor} · {section_path}"
    return anchor
