# Path: app/features/session_memory/domain/enums.py

from enum import Enum


class SessionTurnMode(str, Enum):
    """Which chat entrypoint answered a turn. Both share one conversation history."""
    CHAT = "chat"
    DEEP_RESEARCH = "deep_research"
