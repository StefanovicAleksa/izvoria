# Canvas: context_unification_entities_v1
# Path: app/features/context_unification/domain/entities.py

import uuid
from typing import List
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict

from app.features.visual_context_synthesis.domain.entities import VisualContextFrame


class UnifiedChunk(BaseModel):
    """
    The final, fully prepared data payload representing a specific segment of the media.
    Contains the deterministically stitched string of audio and visual context,
    along with the exact timestamps and image pointers needed for RAG indexing and UI rendering.
    """
    chunk_id: uuid.UUID

    # Exact time bounds from the audio transcript
    start_sec: float
    end_sec: float

    # The stitched string: [Audio Context] + [Visual Context] + Raw Text
    text_payload: str = Field(description="The fully assembled string ready for vector embedding.")

    # Frame pointers for the UI to display the exact images associated with this chunk
    visual_evidence: List[VisualContextFrame] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True)


class UnifiedTranscript(BaseModel):
    """
    The Aggregate Root for Phase 15.
    Represents the complete, RAG-ready document for a given Source ID.
    """
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    source_id: uuid.UUID

    chunks: List[UnifiedChunk] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(from_attributes=True)