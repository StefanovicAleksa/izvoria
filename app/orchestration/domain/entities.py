# filename: orchestration_entities_v1.py
import uuid
from typing import List, Optional, Any
from pydantic import BaseModel, Field, ConfigDict

from app.orchestration.domain.enums import PipelinePhase, PhaseStatus


# --- Individual Job Results ---

class ScanJobResult(BaseModel):
    """Result of the Phase 1: Directory Scan."""
    target_directory: str
    total_found: int
    file_paths: List[str]

    model_config = ConfigDict(frozen=True)


class IngestJobResult(BaseModel):
    """Result of a single file in Phase 2: Ingest."""
    file_path: str
    success: bool
    source_id: Optional[uuid.UUID] = None
    error_message: Optional[str] = None

    model_config = ConfigDict(frozen=True)


class AudioExtractionJobResult(BaseModel):
    """Result of a single file in Phase 3: Audio Extraction."""
    source_id: uuid.UUID
    success: bool
    extracted_file_id: Optional[uuid.UUID] = None
    error_message: Optional[str] = None

    model_config = ConfigDict(frozen=True)


# --- Aggregate Batch Results ---

class PhaseBatchResult(BaseModel):
    """
    Aggregates the results of an entire phase (e.g., all audio extractions)
    so the Pipeline Director knows whether to halt or proceed.
    """
    phase: PipelinePhase
    status: PhaseStatus

    total_processed: int = 0
    successful_count: int = 0
    failed_count: int = 0

    # Can hold a list of IngestJobResult, AudioExtractionJobResult, etc.
    items: List[Any] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True)