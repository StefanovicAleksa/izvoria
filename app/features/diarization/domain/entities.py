# Canvas: diarization_entities_v1
# Path: app/features/diarization/domain/entities.py

import uuid
from typing import List
from datetime import datetime, timezone
from pydantic import BaseModel, ConfigDict, Field


class DiarizationSegment(BaseModel):
    """
    Represents a continuous block of speech attributed to a specific speaker.
    Uses seconds (float) to map cleanly against both VAD and Transcription segments.
    """
    start_sec: float
    end_sec: float
    speaker_id: str = Field(description="The generic label assigned by the model (e.g., 'Speaker_0', 'Speaker_1').")

    model_config = ConfigDict(frozen=True)


class DiarizationMap(BaseModel):
    """
    The complete virtual map of all speakers in a given media source.
    This will be passed to our alignment utility to update the Transcript.
    """
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    source_id: uuid.UUID
    segments: List[DiarizationSegment]
    # True when clustering failed and every speech segment was attributed to a single speaker.
    fallback_applied: bool = Field(default=False)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def unique_speakers(self) -> List[str]:
        """Convenience property to list all distinct speakers identified in the source."""
        return list({seg.speaker_id for seg in self.segments})