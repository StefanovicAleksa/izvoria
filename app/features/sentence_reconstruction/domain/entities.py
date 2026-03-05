# Canvas: sentence_entities_v4
# Path: app/features/sentence_reconstruction/domain/entities.py

import uuid
from typing import List
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict

from app.features.transcription.domain.entities import TranscriptSegment


class ReconstructedSentence(BaseModel):
    """
    A grammatically complete sentence mapped to underlying audio segments.
    """
    sentence_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    text: str

    # Strictly enforced: If Diarization fails, this phase must also fail.
    speaker_id: str

    start_sec: float
    end_sec: float

    # Keeps an absolute audit trail back to the exact acoustic Whisper tokens
    segments: List[TranscriptSegment]

    model_config = ConfigDict(frozen=True)


class ReconstructedTranscript(BaseModel):
    """
    The aggregate root representing the fully processed video transcript.
    """
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    source_id: uuid.UUID
    sentences: List[ReconstructedSentence] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Required to deserialize from SQLAlchemy DB models
    model_config = ConfigDict(from_attributes=True)