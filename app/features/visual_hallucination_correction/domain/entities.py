# Canvas: visual_hallucination_entities_v2
# Path: app/features/visual_hallucination_correction/domain/entities.py

import uuid
from typing import List, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict


class VisualHallucinationAudit(BaseModel):
    """
    Represents the SLM's strict evaluation of a visual description against its source text chunk.
    Acts purely as an immutable ledger recording whether the visual data was kept or dropped.
    """
    chunk_id: uuid.UUID = Field(description="The UUID of the target ContextualizedChunk.")

    is_visual_accepted: bool = Field(default=False,
                                     description="True if the visual description semantically matched the text.")

    discard_reason: Optional[str] = Field(
        default=None,
        description="The strict 1-sentence reason the SLM rejected the visual data, if applicable."
    )

    model_config = ConfigDict(frozen=True)


class VisualHallucinationEvaluatedTranscript(BaseModel):
    """
    The Aggregate Root for the Visual Hallucination Correction phase.
    Maintains the complete audit trail of accepted/rejected visuals for a specific source.
    """
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    source_id: uuid.UUID = Field(description="The UUID of the original Video Source.")

    audits: List[VisualHallucinationAudit] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(from_attributes=True)