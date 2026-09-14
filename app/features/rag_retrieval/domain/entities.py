# Canvas: rag_retrieval_entities_v3
# Path: app/features/rag_retrieval/domain/entities.py

import uuid
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


class SearchQuery(BaseModel):
    """Represents the incoming search request from the user/UI."""
    prompt: str = Field(..., min_length=1, description="The raw text query from the user.")
    limit: int = Field(default=10, ge=1, le=20, description="Maximum number of results to return.")

    # Optional filters to restrict search to a specific case/patient (Source)
    source_id_filter: Optional[uuid.UUID] = Field(default=None,
                                                  description="Optional: Restrict search to a specific source.")

    # Whitespace is stripped before min_length is checked, so blank prompts are rejected
    # here instead of failing deep inside the embedder with a model already loaded.
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)


class RetrievedChunk(BaseModel):
    """Represents a single contextualized chunk retrieved from the hybrid search."""
    chunk_id: uuid.UUID
    source_id: uuid.UUID

    # Contextualized text (generated context + transcript) that was embedded and keyword-indexed
    text_payload: str

    # Verbatim speaker-labelled transcript text: what an answer model is allowed to read and cite
    raw_text: str = Field(default="", description="Verbatim speaker/timestamp-labelled transcript text.")
    speaker_labels: List[str] = Field(default_factory=list)

    start_sec: float
    end_sec: float

    # Reciprocal Rank Fusion (RRF) combined score
    relevance_score: float = Field(..., description="The unified score from Dense + Sparse search.")

    # Cross-encoder score, set when the chunk passed through the reranker
    rerank_score: Optional[float] = Field(default=None, description="Cross-encoder relevance score.")

    # The database linkage to the Storage Feature (Onyx Vault)
    extracted_file_id: Optional[uuid.UUID] = Field(default=None, description="Link to the physical .mp4 clip.")

    model_config = ConfigDict(frozen=True)


class SearchResult(BaseModel):
    """The final payload returned to the orchestrator or frontend API."""
    query: str
    results: List[RetrievedChunk]
    execution_time_ms: int = Field(default=0)

    model_config = ConfigDict(frozen=True)
