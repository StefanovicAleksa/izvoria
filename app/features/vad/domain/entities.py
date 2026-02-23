# File: app/features/vad/domain/entities.py
import uuid
from typing import List
from datetime import datetime, timezone
from pydantic import BaseModel, ConfigDict, Field


class VadSegment(BaseModel):
    """
    Represents a single continuous block of human speech.
    Uses seconds (float) to easily interface with PyTorch/Numpy slicing.
    """
    start_sec: float
    end_sec: float
    confidence: float = Field(ge=0.0, le=1.0, description="AI confidence score that this is speech.")

    model_config = ConfigDict(frozen=True)


class VadMap(BaseModel):
    """
    The complete virtual map of all speech in a given media source.
    This acts as the Rosetta Stone for Transcription and Diarization.
    """
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    source_id: uuid.UUID
    segments: List[VadSegment]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def total_speech_seconds(self) -> float:
        """Calculates the total duration of actual human speech."""
        return sum(seg.end_sec - seg.start_sec for seg in self.segments)