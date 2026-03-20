# Canvas: sentence_entities_v5
# Path: app/features/sentence_reconstruction/domain/entities.py

import uuid
from typing import List
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict

# We import TranscriptWord because it is now our atomic building block
from app.features.transcription.domain.entities import TranscriptWord


class PunctuatedWord(BaseModel):
    """
    A transient data structure used only during the Phase 7 pipeline execution.
    It pairs the original, mathematically perfect acoustic word with the
    punctuation tag predicted by the Token Classifier (e.g., '.', '?', ',', or '0' for NONE).
    """
    acoustic_word: TranscriptWord
    punctuation_tag: str = Field(description="The strict punctuation class predicted by the model.")

    model_config = ConfigDict(frozen=True)


class ReconstructedSentence(BaseModel):
    """
    A grammatically complete sentence built deterministically.
    Instead of tracking high-level segments, it now tracks the exact array of words
    that form the sentence, guaranteeing a flawless audit trail for the video player UI.
    """
    sentence_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    text: str = Field(description="The fully assembled string (words + injected punctuation).")

    # Strictly enforced: If Diarization fails, this phase must also fail.
    speaker_id: str

    start_sec: float
    end_sec: float

    # NEW: The absolute acoustic audit trail replaces the old 'segments' array
    words: List[TranscriptWord]

    model_config = ConfigDict(frozen=True)


class ReconstructedTranscript(BaseModel):
    """
    The aggregate root representing the fully processed video transcript.
    This acts as the final RAG payload (once Phase 8 phonetic correction passes over it).
    """
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    source_id: uuid.UUID
    sentences: List[ReconstructedSentence] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Required to deserialize from SQLAlchemy DB models
    model_config = ConfigDict(from_attributes=True)