# Canvas: transcription_entities_v3
# Path: app/features/transcription/domain/entities.py

import uuid
from typing import List, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, ConfigDict, Field


class TranscriptWord(BaseModel):
    """
    Represents a single word with absolute timestamps.
    Crucial for Zero-Math deterministic sentence reconstruction.
    """
    start_sec: float
    end_sec: float
    word: str
    probability: float = Field(default=1.0, description="Confidence score for this specific word.")

    model_config = ConfigDict(frozen=True)


class TranscriptSegment(BaseModel):
    """
    Represents a single block of transcribed text corresponding to a specific time range.
    """
    start_sec: float
    end_sec: float
    text: str = Field(description="The actual transcribed text for this audio segment.")

    # NEW: The precise word-level breakdown of the text
    words: List[TranscriptWord] = Field(default_factory=list)

    # Optional field for future-proofing when we add the Diarization feature
    speaker_id: Optional[str] = None

    # NEW: Whisper Engine Metadata for Zero-VRAM Hallucination Scrubbing
    no_speech_prob: Optional[float] = None
    avg_logprob: Optional[float] = None
    compression_ratio: Optional[float] = None

    model_config = ConfigDict(frozen=True)


class Transcript(BaseModel):
    """
    The complete textual representation of a media source.
    Acts as the primary text payload for the RAG Vector Database.
    """
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    source_id: uuid.UUID = Field(description="The UUID of the original media source.")
    segments: List[TranscriptSegment]

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def full_text(self) -> str:
        """Convenience property to get the entire transcript as a single string."""
        return " ".join(seg.text.strip() for seg in self.segments)

    model_config = ConfigDict(from_attributes=True)