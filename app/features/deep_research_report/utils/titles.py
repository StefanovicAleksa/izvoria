# Path: app/features/deep_research_report/utils/titles.py

"""
A report's title, tidied the same way wherever it comes from: the scoping model, the outline editor or a rename.
"""

import re

MAX_TITLE_CHARS = 120

# Wrapping a model might add around a title: Markdown heading marks, quotes, a "Title:" label
_HEADING = re.compile(r"^#+\s*")
_LABEL = re.compile(r"^title\s*:\s*", re.IGNORECASE)
_QUOTES = "\"'“”„‘’«»`*_"


def clean_title(text: object) -> str:
    """One line, no heading marks, wrapping quotes or trailing full stop, at most MAX_TITLE_CHARS; "" if empty."""
    title = " ".join(str(text or "").split())
    title = _LABEL.sub("", _HEADING.sub("", title))
    title = title.strip(_QUOTES + " ")
    # A trailing full stop or colon is a sentence's, not a title's; "?" and "!" are kept
    title = title.rstrip(".:;, ")
    if len(title) > MAX_TITLE_CHARS:
        title = title[:MAX_TITLE_CHARS].rsplit(" ", 1)[0].rstrip(".:;,- ") or title[:MAX_TITLE_CHARS]
    return title
