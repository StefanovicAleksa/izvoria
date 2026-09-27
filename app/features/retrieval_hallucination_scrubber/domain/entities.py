# Canvas: verified_retrieval_entities_v2
# Path: app/features/retrieval_hallucination_scrubber/domain/entities.py

import uuid
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class ScrubAudit(BaseModel):
    """
    Represents the SLM's binary decision on whether a retrieved RAG chunk
    is genuinely relevant to the user's prompt.
    """
    chunk_id: uuid.UUID = Field(description="The UUID of the retrieved vector chunk.")

    is_relevant: bool = Field(
        default=False,
        description="True if the chunk contains information that helps answer the prompt."
    )

    discard_reason: Optional[str] = Field(
        default=None,
        description="The strict 1-sentence reason the SLM rejected the chunk, if applicable."
    )

    evaluation_failed: bool = Field(
        default=False,
        description="True if the SLM produced an unparseable verdict and the chunk was kept by default."
    )

    kept_by_rerank_score: bool = Field(
        default=False,
        description="True if the chunk skipped the SLM: the cross-encoder scored it at or above "
                    "RETRIEVAL_SCRUBBER_TRUST_RERANK_SCORE."
    )

    model_config = ConfigDict(frozen=True)
