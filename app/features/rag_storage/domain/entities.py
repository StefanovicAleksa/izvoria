# Canvas: rag_storage_entities_v1
# Path: app/features/rag_storage/domain/entities.py

from dataclasses import dataclass
from typing import Optional
from uuid import UUID

@dataclass
class IndexingTelemetry:
    """Tracks the performance and statistical outcome of an ingestion run."""
    source_id: UUID
    total_chunks_processed: int
    dense_vectors_saved: int
    sparse_vectors_saved: int
    processing_time_seconds: float
    embedding_model_used: str

@dataclass
class IngestionResult:
    """The final result returned to the Orchestrator after storage is complete."""
    success: bool
    telemetry: IndexingTelemetry
    error_message: Optional[str] = None

@dataclass(frozen=True)
class StoredChunkText:
    """An indexed chunk's texts as stored: its generated context (None for a chunk indexed without one) and its
    verbatim raw text."""
    chunk_id: UUID
    context_text: Optional[str]
    raw_text: str
