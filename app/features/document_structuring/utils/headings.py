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
_NUMBER_THEN_PERIOD = re.compile(r"(?:\d|\b[IVXLCDM]+)\.$")
# Longer than this, a line is a sentence that happens to start with a number or a capital
MAX_HEADING_WORDS = 12
# A line in capitals that opens a quotation is shouted dialogue ("“BALLARAT.”"), never a heading. Every opening
# mark the three packs' languages use: English “ ‘ ", Serbian/Slovenian „ ‚ » «.
_OPENING_QUOTES = ("“", "‘", '"', "'", "„", "‚", "»", "«")

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
    if (len(letters) >= 4 and len(words) <= MAX_HEADING_WORDS and all(c.isupper() for c in letters)
            and not stripped.startswith(_OPENING_QUOTES) and not _ends_a_statement(stripped)):
        return TITLE_LEVEL, stripped

    return None


def _ends_a_sentence(line: str) -> bool:
    # "4.2." and a book's "CHAPTER IV." end in a period without being sentences
    return line.endswith((".", ";", ",", ":")) and not _NUMBER_THEN_PERIOD.search(line)


def _ends_a_statement(line: str) -> bool:
    """
    A line in capitals that ends in a full stop, "!" or "?" -- its closing quote aside -- is a sentence: a word
    of dialogue, or the tail of a licence's shouted disclaimer ("DAMAGE."). A title doesn't end like that, but
    "CHAPTER IV." does. Colons stay: "RECITALS:" is a heading.
    """
    bare = line.rstrip("”’\"'»«")
    if bare.endswith((",", ";")):
        # A wrapped line of a shouted paragraph ("…FOR ANY INDIRECT,"): a title never ends mid-sentence
        return True
    return bare.endswith((".", "!", "?")) and not _NUMBER_THEN_PERIOD.search(bare)


def is_capitals_title(line: str, clause_patterns: List[Pattern]) -> bool:
    """A heading only because it is written in capitals: no Markdown, clause marker or number says so."""
    stripped = line.strip()
    return (heading_level(stripped, clause_patterns) is not None and not _MARKDOWN_HEADING.match(stripped)
            and not any(pattern.match(stripped) for pattern in clause_patterns) and not _NUMBERED.match(stripped))


def ends_a_line_of_thought(line: str) -> bool:
    """The line closes its sentence (or introduces a list), so what follows it can start something new."""
    return line.rstrip().rstrip("”’\"'»«").endswith((".", "!", "?", ":"))
