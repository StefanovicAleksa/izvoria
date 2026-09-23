# Canvas: context_unification_entities_v2
# Path: app/features/context_unification/domain/entities.py

import uuid
from typing import List, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict

from app.core.anchors.domain.enums import AnchorKind
from app.features.visual_context_synthesis.domain.entities import VisualContextFrame


class UnifiedChunk(BaseModel):
    """
    The final, fully prepared data payload representing a specific segment of the media.
    Contains the deterministically stitched string of audio and visual context,
    along with the exact timestamps and image pointers needed for RAG indexing and UI rendering.
    """
    chunk_id: uuid.UUID

    # Exact time bounds from the audio transcript; 0.0 for document chunks (see anchor_kind)
    start_sec: float
    end_sec: float

    # Where a citation of this chunk points: a time range, a page range, or a document section
    anchor_kind: AnchorKind = AnchorKind.TIME
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section_path: Optional[str] = None

    # The stitched string: [Audio Context] + [Visual Context] + Raw Text
    text_payload: str = Field(description="The fully assembled string ready for vector embedding.")

    # Verbatim, speaker/timestamp-labelled transcript text. The only text an answer model may cite.
    raw_text: str = Field(default="", description="Verbatim speaker-labelled transcript text of the chunk.")

    # Model-generated situating context (audio + visual). Indexed for retrieval, never citable evidence.
    context_text: str = Field(default="", description="SLM/VLM-generated context prepended for retrieval only.")

    speaker_labels: List[str] = Field(default_factory=list, description="Speakers in order of first appearance.")

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
