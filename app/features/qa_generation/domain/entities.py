# Canvas: qa_generation_entities_v1
# Path: app/features/qa_generation/domain/entities.py

import uuid
from typing import Dict, List
from pydantic import BaseModel, Field, ConfigDict


class QACitation(BaseModel):
    """
    Represents a specific citation linking generated text back to a physical .mp4 clip.
    """
    doc_id: str = Field(..., description="The strict citation tag used by the LLM (e.g., '[Doc 1]').")
    chunk_id: uuid.UUID = Field(..., description="The database ID of the retrieved chunk.")
    physical_path: str = Field(..., description="The local OS path to the extracted .mp4 clip.")
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