# Canvas: hallucination_scrubber_entities_v1
# Path: app/features/hallucination_scrubber/domain/entities.py

import uuid
from typing import List
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict

from app.features.transcription.domain.entities import TranscriptSegment


class ScrubbingPatch(BaseModel):
    """
    Represents a specific sub-clause or text block that the AI identified as a hallucination.
    Read-only record used for compliance auditing.
    """
    removed_text: str = Field(description="The exact string that was deleted.")
    detected_label: str = Field(description="The semantic category (e.g., 'youtube outro') that triggered the removal.")
    confidence_score: float = Field(description="The probability score output by the Zero-Shot model.")

    model_config = ConfigDict(frozen=True)


class AppliedScrubbingRecord(BaseModel):
    """
    An immutable audit record proving exactly what the Orchestrator removed from a specific segment.
    Since raw segments lack UUIDs at this pipeline stage, we use the start time as the identifier.
    """
    segment_start_sec: float
    segment_end_sec: float
    original_text: str
    scrubbed_text: str
    applied_patches: List[ScrubbingPatch] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True)


class HallucinationScrubbedTranscript(BaseModel):
    """
    The Final Aggregate Root for Phase 6.5.
    Contains the cleaned transcription segments and the cryptographic-style audit log of all deletions.
    """
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    source_id: uuid.UUID = Field(description="The UUID of the original media source.")

    segments: List[TranscriptSegment] = Field(default_factory=list)
    audit_trail: List[AppliedScrubbingRecord] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(from_attributes=True)