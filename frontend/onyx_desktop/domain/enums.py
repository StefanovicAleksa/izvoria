# Canvas: desktop_enums_v1
# Path: frontend/onyx_desktop/domain/enums.py

from enum import Enum


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMode(str, Enum):
    """Which engine call a chat turn maps to."""
    SINGLE_PASS = "single_pass"      # OnyxEngine.chat()
    DEEP_RESEARCH = "deep_research"  # OnyxEngine.chat_deep_research()


class SearchMode(str, Enum):
    STANDARD = "standard"   # OnyxEngine.search_standard()
    VERIFIED = "verified"   # OnyxEngine.search_verified()


class IngestMode(str, Enum):
    VIDEO = "video"   # OnyxEngine.ingest_videos()
    AUDIO = "audio"   # OnyxEngine.ingest_audios()


class ApplianceStatus(str, Enum):
    """
    What the status bar shows. Mirrors EngineState, plus WORKING for a live call in flight:
    the engine stays ONLINE during search/chat, but the UI must not accept a second request
    while one is running (the VRAM pool holds a single model at a time).
    """
    OFFLINE = "offline"
    BOOTING = "booting"
    READY = "ready"
    WORKING = "working"
    INGESTING = "ingesting"
    ERROR = "error"


class MediaKind(str, Enum):
    """
    What the player was handed. A citation's path is a per-chunk clip when the chunk had one
    extracted, otherwise the whole original source file -- which decides whether the player
    seeks to the chunk's timestamp or just plays from the start.
    """
    CLIP = "clip"                # the file IS the chunk; play from 0
    FULL_SOURCE = "full_source"  # seek to start_sec, stop at end_sec
    UNRESOLVED = "unresolved"    # no physical path came back


class TurnOutcome(str, Enum):
    """
    Deep Research returns either an answer or a question back at the user, and the
    clarification branch is the common case rather than an edge case.
    """
    ANSWERED = "answered"
    NEEDS_CLARIFICATION = "needs_clarification"
    FAILED = "failed"


class JobProgressState(str, Enum):
    """Mirrors what JobManager.run_job reports around each per-file unit of work."""
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
