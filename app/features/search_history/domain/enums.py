# Path: app/features/search_history/domain/enums.py

from enum import Enum


class SearchHistoryMode(str, Enum):
    """Which search produced a record: plain hybrid search, or the hallucination-scrubbed one."""
    STANDARD = "standard"
    VERIFIED = "verified"
