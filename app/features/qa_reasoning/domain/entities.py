# Path: app/features/qa_reasoning/domain/entities.py

import uuid
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict

from app.features.qa_reasoning.domain.enums import ReasonerDecisionType


class EvidenceLink(BaseModel):
    """Ties one claim in the Reasoner's interim analysis to the specific chunk that supports it."""
    chunk_id: uuid.UUID = Field(description="The evidence_pool chunk this claim is grounded in.")
    claim: str = Field(description="The specific claim the chunk supports.")

    model_config = ConfigDict(frozen=True)


class EvidenceBlock(BaseModel):
    """The Reasoner's formatted evidence prompt section and the map from its [E<n>] tags to chunk ids."""
    formatted_text: str
    tag_mapping: Dict[str, uuid.UUID] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class ReasonerDecision(BaseModel):
    """
    One turn of the Reasoner's chain-of-thought over the accumulated evidence pool: an
    evidence-linked interim analysis plus exactly one control-flow decision for the pipeline.
    """
    decision: ReasonerDecisionType

    chain_of_thought: str = Field(
        description="The Reasoner's step-by-step reasoning. Kept in the trace only, never shown raw to the user."
    )
    interim_analysis: str = Field(
        description="Evidence-linked prose analysis of what the evidence pool supports so far."
    )
    evidence_links: List[EvidenceLink] = Field(default_factory=list)

    refined_query: Optional[str] = Field(
        default=None, description="Set iff decision is REFINE_QUERY: the rewritten/narrower search query."
    )
    clarification_question: Optional[str] = Field(
        default=None, description="Set iff decision is ASK_CLARIFICATION: the question to ask the user back."
    )

    model_config = ConfigDict(frozen=True)
