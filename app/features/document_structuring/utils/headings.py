# Path: app/features/document_structuring/utils/headings.py

"""
Is this line a heading, and how deep? Text layers and OCR output carry no heading markup, so a heading is
recognised from its wording: a Markdown heading (DOCX), a clause marker from the language pack ("Article 7",
"3. člen"), a short numbered title ("4.2 Payment Terms"), or a short line in capitals ("LEASE AGREEMENT").
"""

import re
from typing import List, Optional, Pattern, Tuple

_MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
_NUMBERED = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+\S")
# Longer than this, a line is a sentence that happens to start with a number or a capital
MAX_HEADING_WORDS = 12

TITLE_LEVEL = 1
CLAUSE_LEVEL = 2


def compile_clause_patterns(patterns: List[str]) -> List[Pattern]:
    return [re.compile(pattern) for pattern in patterns]


def heading_level(line: str, clause_patterns: List[Pattern]) -> Optional[Tuple[int, str]]:
    """(level, heading text) when `line` is a heading, else None. The text drops Markdown's '#' markup only."""
    stripped = line.strip()
    if not stripped or stripped.startswith("|"):
        return None

    markdown = _MARKDOWN_HEADING.match(stripped)
    if markdown:
        return len(markdown.group(1)), markdown.group(2).strip()

    words = stripped.split()
    if any(pattern.match(stripped) for pattern in clause_patterns):
        # "Article 7 Termination" is a heading; "Article 7 applies to every lease signed after..." is not
        if len(words) <= MAX_HEADING_WORDS and not _ends_a_sentence(stripped):
            return CLAUSE_LEVEL, stripped
        return None

    numbered = _NUMBERED.match(stripped)
    if numbered and len(words) <= MAX_HEADING_WORDS and not _ends_a_sentence(stripped):
        depth = numbered.group(1).count(".") + 1
        return CLAUSE_LEVEL + depth - 1, stripped

    letters = [c for c in stripped if c.isalpha()]
    if len(letters) >= 4 and len(words) <= MAX_HEADING_WORDS and all(c.isupper() for c in letters):
        return TITLE_LEVEL, stripped

    return None


def _ends_a_sentence(line: str) -> bool:
    return line.endswith((".", ";", ",", ":")) and not re.search(r"\d\.$", line)
