# Path: app/features/visual_context_synthesis/utils/narrative_guard.py

import re
from typing import List

_NUMBER = re.compile(r"\d+(?:[.,:/-]\d+)*")
_QUOTED = re.compile(r'"([^"]+)"|“([^”]+)”')
FALLBACK_MAX_CHARS = 700


def _normalise(text: str) -> str:
    return " ".join(text.casefold().split())


def narrative_is_supported(narrative: str, sources: List[str]) -> bool:
    """
    True if every number and every quoted string in the synthesized narrative appears in the frame descriptions it
    was written from. A small model merging a timeline tends to add values ("160.84" from nowhere) or requote
    on-screen text loosely; either would be indexed as visual evidence.
    """
    source_text = _normalise(" ".join(sources))
    source_numbers = {n.strip(".,:/-") for n in _NUMBER.findall(source_text)}
    if any(n.strip(".,:/-") not in source_numbers for n in _NUMBER.findall(narrative)):
        return False
    return all(_normalise(a or b) in source_text for a, b in _QUOTED.findall(narrative))


def fallback_narrative(sources: List[str], max_chars: int = FALLBACK_MAX_CHARS) -> str:
    """The distinct frame descriptions joined in order, cut at the last whole sentence within `max_chars`."""
    joined = " ".join(dict.fromkeys(s.strip() for s in sources if s.strip()))
    if len(joined) <= max_chars:
        return joined
    cut = joined[:max_chars]
    end = cut.rfind(". ")
    return cut[:end + 1] if end > 0 else cut.rstrip() + "…"
