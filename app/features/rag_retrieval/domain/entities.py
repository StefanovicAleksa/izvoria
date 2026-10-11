# Canvas: rag_retrieval_entities_v3
# Path: app/features/rag_retrieval/domain/entities.py

import uuid
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict

from app.core.anchors.domain.enums import AnchorKind


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

    # Time range for transcript chunks; 0.0 for document chunks, which carry a page range or section instead
    start_sec: float
    end_sec: float
    anchor_kind: AnchorKind = AnchorKind.TIME
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section_path: Optional[str] = None

    # Order of the chunk within its source; reading order and neighbour lookup use it, never start_sec
    position: int = 0

    # Reciprocal Rank Fusion (RRF) combined score
    relevance_score: float = Field(..., description="The unified score from Dense + Sparse search.")

    # Cross-encoder score, set when the chunk passed through the reranker
    rerank_score: Optional[float] = Field(default=None, description="Cross-encoder relevance score.")

    # The database linkage to the Storage Feature (Izvoria Vault)
    extracted_file_id: Optional[uuid.UUID] = Field(default=None, description="Link to the physical .mp4 clip.")

    model_config = ConfigDict(frozen=True)


class SearchResult(BaseModel):
    """The final payload returned to the orchestrator or frontend API."""
    query: str
    results: List[RetrievedChunk]
    execution_time_ms: int = Field(default=0)

    model_config = ConfigDict(frozen=True)


class IndexedSource(BaseModel):
    """
    A source that actually has rows in `vector_chunks`.

    Distinct from storage's `Source`, which lists what was *ingested*: a file can be ingested and still have
    nothing indexed (ingestion failed partway, or it held no speech). Anything that wants to reason about
    corpus coverage needs the indexed set, not the ingested one.
    """
    source_id: uuid.UUID
    name: str
    chunk_count: int
    first_start_sec: float = 0.0
    last_end_sec: float = 0.0
    # A document's kind and length, so a prompt can say "12 pages" instead of "about 00:00:00 long"
    anchor_kind: AnchorKind = AnchorKind.TIME
    page_count: Optional[int] = None

    model_config = ConfigDict(frozen=True)


class CorpusSweep(BaseModel):
    """
    A per-document sweep: run `prompt` against each of the best-matching indexed sources individually, so a
    relevant recording is still found when no single query ranked it into the global top-k.

    `ann_depth` and `max_sources` are the cost gate. Picking sweep sources by grouping over all of
    `vector_chunks` would be a full scan with a distance computation per row; one index-served ANN probe
    grouped down to its best sources costs the same whether the corpus holds ten thousand chunks or ten
    million.
    """
    prompt: str = Field(..., min_length=1)
    per_source_limit: int = Field(default=3, ge=1, le=20)
    max_sources: int = Field(default=50, ge=1)
    ann_depth: int = Field(default=2000, ge=1)

    model_config = ConfigDict(frozen=True)


class BatchSearchResult(BaseModel):
    """
    One embedder acquisition's worth of searching: the fan-out's per-query results plus the sweep's hits.

    Kept apart rather than merged because they mean different things downstream -- a fan-out hit is
    attributable to the sub-question that found it, a sweep hit is attributable to nothing but the topic.
    """
    results: List[SearchResult] = Field(default_factory=list)
    sweep_results: List[RetrievedChunk] = Field(default_factory=list)
    swept_source_count: int = 0
    execution_time_ms: int = 0
