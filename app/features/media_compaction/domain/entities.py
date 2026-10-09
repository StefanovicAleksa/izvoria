import uuid
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict


class MediaProbe(BaseModel):
    """What ffprobe says about a recording: its first real video and audio streams, and its length."""
    video_codec: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    audio_codec: Optional[str] = None
    # Bits per second of the audio stream; for an audio-only file whose stream doesn't say, of the whole file
    audio_bit_rate: Optional[int] = None
    # Bits per second of the whole file
    bit_rate: Optional[int] = None
    duration_sec: Optional[float] = None

    model_config = ConfigDict(frozen=True)


class CompactionStatus(str, Enum):
    REPLACED = "replaced"                # the original was swapped for the smaller copy
    ALREADY_COMPACT = "already_compact"  # nothing to gain (e.g. a 720p H.264 YouTube download)
    KEPT_ORIGINAL = "kept_original"      # a copy was made but failed a guard, so the original stays
    NOT_MEDIA = "not_media"              # documents and images are kept as they are
    PENDING = "pending"                  # a dry run: this original would be re-encoded


class CompactionOutcome(BaseModel):
    source_id: uuid.UUID
    status: CompactionStatus
    bytes_before: int = 0
    bytes_after: int = 0
    reason: Optional[str] = None

    model_config = ConfigDict(frozen=True)

    @property
    def bytes_saved(self) -> int:
        return max(0, self.bytes_before - self.bytes_after) if self.status is CompactionStatus.REPLACED else 0
