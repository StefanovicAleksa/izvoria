# Canvas: desktop_formatting_v1
# Path: frontend/onyx_desktop/utils/formatting.py

"""Display helpers. Pure functions, no Qt, so they are cheap to unit-test."""

import re
from typing import Iterable, List, Optional, Tuple

_DOC_TAG = re.compile(r"\[Doc (\d+)]")


def timestamp(seconds: float) -> str:
    """`mm:ss`, or `h:mm:ss` past an hour. Negative and NaN-ish inputs clamp to 0."""
    try:
        total = max(0, int(seconds))
    except (TypeError, ValueError):
        total = 0
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def timespan(start_sec: float, end_sec: float) -> str:
    return f"{timestamp(start_sec)}–{timestamp(end_sec)}"


def location(evidence) -> str:
    """Where an excerpt sits: a document's page and section, or a recording's time range."""
    return evidence.location or timespan(evidence.start_sec, evidence.end_sec)


def duration(seconds: float) -> str:
    """Human wall-clock for an ingestion report: `4m 12s`, `1h 03m`."""
    try:
        total = max(0, int(seconds))
    except (TypeError, ValueError):
        total = 0
    if total < 60:
        return f"{total}s"
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m {secs:02d}s"


def score(value: Optional[float]) -> str:
    """Scores are only meaningful on ranked results; a hydration read has none."""
    return "—" if value is None else f"{value:.3f}"


def speaker_summary(speakers: Iterable[str]) -> str:
    unique = list(dict.fromkeys(s for s in speakers if s))
    if not unique:
        return ""
    if len(unique) <= 2:
        return ", ".join(unique)
    return f"{unique[0]}, {unique[1]} +{len(unique) - 2}"


def split_doc_tags(text: str) -> List[Tuple[str, Optional[int]]]:
    """
    Splits answer text into `(segment, doc_number)` runs so the chat bubble can render each
    `[Doc N]` as a clickable link and the prose around it as plain text.

    Assumes `text` has already been citation-normalized by the gateway: the engine returns the
    raw model string, which can still contain grouped tags like `[Doc 1, 3]` that this would
    otherwise miss.
    """
    parts: List[Tuple[str, Optional[int]]] = []
    cursor = 0
    for match in _DOC_TAG.finditer(text):
        if match.start() > cursor:
            parts.append((text[cursor:match.start()], None))
        parts.append((match.group(0), int(match.group(1))))
        cursor = match.end()
    if cursor < len(text):
        parts.append((text[cursor:], None))
    return parts


def elide(text: str, limit: int = 160) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "…"
