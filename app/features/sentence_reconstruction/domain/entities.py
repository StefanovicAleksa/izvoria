# Canvas: sentence_entities_v6
# Path: app/features/sentence_reconstruction/domain/entities.py

import uuid
from typing import List
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict

from app.features.transcription.domain.entities import TranscriptWord


class PunctuatedWord(BaseModel):
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

    # Adding default="" registers the attribute at the class level for the IDE
    text: str = Field(default="", description="The fully assembled string (words + injected punctuation).")

    # Strictly enforced: If Diarization fails, this phase must also fail.
    speaker_id: str = Field(default="Unknown", description="The assigned speaker ID from Diarization.")

    start_sec: float = Field(default=0.0)
    end_sec: float = Field(default=0.0)

    # The absolute acoustic audit trail replaces the old 'segments' array
    words: List[TranscriptWord] = Field(default_factory=list)

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

    model_config = ConfigDict(from_attributes=True)