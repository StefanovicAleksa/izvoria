# Canvas: desktop_formatting_v1
# Path: frontend/izvoria_desktop/utils/formatting.py

"""Display helpers. Pure functions, no Qt, so they are cheap to unit-test."""

import re
from datetime import datetime
from typing import Iterable, List, Optional, Tuple

from izvoria_desktop.utils.i18n import N_, tr

_DOC_TAG = re.compile(r"\[Doc (\d+)]")

_MONTHS = (N_("Jan"), N_("Feb"), N_("Mar"), N_("Apr"), N_("May"), N_("Jun"),
           N_("Jul"), N_("Aug"), N_("Sep"), N_("Oct"), N_("Nov"), N_("Dec"))


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
        return tr("{s}s", s=total)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return tr("{h}h {m}m", h=hours, m=f"{minutes:02d}")
    return tr("{m}m {s}s", m=minutes, s=f"{secs:02d}")


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


def when(moment: Optional[datetime], now: Optional[datetime] = None) -> str:
    """
    A history row's date, as short as stays unambiguous, in local time: "14:32" today,
    "Yesterday 14:32", "12 Sep 14:32" this year, "12 Sep 2025" before that.
    """
    if moment is None:
        return ""
    local = moment.astimezone() if moment.tzinfo else moment
    now = (now or datetime.now(local.tzinfo)).astimezone(local.tzinfo) if local.tzinfo else (now or datetime.now())
    days = (now.date() - local.date()).days
    clock = local.strftime("%H:%M")
    if days == 0:
        return clock
    if days == 1:
        return tr("Yesterday {time}", time=clock)
    # Not strftime('%b'): that follows the process locale, not the display language
    month = tr(_MONTHS[local.month - 1])
    if local.year == now.year:
        return tr("{day} {month} {time}", day=local.day, month=month, time=clock)
    return tr("{day} {month} {year}", day=local.day, month=month, year=local.year)
