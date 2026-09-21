# Canvas: qa_generation_entities_v1
# Path: app/features/qa_generation/domain/entities.py

import uuid
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict

from app.features.rag_retrieval.domain.entities import RetrievedChunk
from app.features.qa_generation.domain.enums import CritiqueVerdict


class QACitation(BaseModel):
    """
    Represents a specific citation linking generated text back to a physical .mp4 clip.
    """
    doc_id: str = Field(..., description="The strict citation tag used by the LLM (e.g., '[Doc 1]').")
    chunk_id: uuid.UUID = Field(..., description="The database ID of the retrieved chunk.")
    source_id: uuid.UUID = Field(..., description="The ingested source the chunk came from.")
    extracted_file_id: Optional[uuid.UUID] = Field(
        default=None, description="The vaulted .mp4 clip of this chunk, if one was extracted."
    )
    physical_path: Optional[str] = Field(
        default=None, description="The local OS path to the clip (or source file); resolved by the engine."
    )
    start_sec: float = Field(..., description="Start time of the clip in seconds.")
    end_sec: float = Field(..., description="End time of the clip in seconds.")

    model_config = ConfigDict(frozen=True)


class QAResponse(BaseModel):
    """
    The final payload delivered to the client UI after inference.
    """
    query: str = Field(..., description="The original user prompt.")
    generated_answer: str = Field(..., description="The LLM's final response text with embedded citation tags.")
    citations: List[QACitation] = Field(default_factory=list, description="The mapping of tags to playable video clips.")
    inference_time_ms: int = Field(default=0, description="Total latency for generation.")

    model_config = ConfigDict(frozen=True)


class RAGContextBlock(BaseModel):
    """
    Represents the assembled text block to be injected into the LLM prompt.
    """
    formatted_text: str = Field(..., description="The concatenated string of all surviving chunks with [Doc X] tags.")
    doc_mapping: Dict[str, uuid.UUID] = Field(..., description="Internal map connecting [Doc X] strings to chunk UUIDs.")

    model_config = ConfigDict(frozen=True)


class AnswerCritique(BaseModel):
    """
    The self-critique's verdict on a draft answer: does every claim match the text of the
    chunk it cites, and does the draft fully address the original question?
    """
    verdict: CritiqueVerdict
    gap_description: Optional[str] = Field(
        default=None, description="Set iff verdict is NEEDS_MORE_EVIDENCE: what's missing or unsupported."
    )

    model_config = ConfigDict(frozen=True)

class NumberedChunk(BaseModel):
    """An evidence chunk with the [Doc N] number it carries for the whole conversation."""
    doc_number: int = Field(..., ge=1)
    chunk: RetrievedChunk

    model_config = ConfigDict(frozen=True)


class ConversationTurn(BaseModel):
    """
    An earlier question/answer of a conversation and the evidence first shown in that turn. The answer's
    [Doc N] tags refer to conversation-wide numbers; they are not source material themselves.
    """
    question: str
    answer: str
    evidence: List[NumberedChunk] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True)


class ConversationHistory(BaseModel):
    """Earlier turns, oldest first. New evidence of the next turn is numbered from next_doc_number."""
    turns: List[ConversationTurn] = Field(default_factory=list)
    next_doc_number: int = Field(default=1, ge=1)

    model_config = ConfigDict(frozen=True)


class ConversationAnswer(BaseModel):
    """A conversational answer, plus the chunks it showed for the first time, in [Doc N] order."""
    response: QAResponse
    new_chunk_ids: List[uuid.UUID] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True)
