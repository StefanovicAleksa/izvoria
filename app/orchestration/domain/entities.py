# filename: orchestration_entities_v2.py
# path: app/orchestration/domain/entities.py
import uuid
from typing import List, Optional, Any
from pydantic import BaseModel, Field, ConfigDict

from app.orchestration.domain.enums import PipelinePhase, PhaseStatus


# --- Individual Job Results ---

class ScanJobResult(BaseModel):
    target_directory: str
    total_found: int
    file_paths: List[str]

    model_config = ConfigDict(frozen=True)


class IngestJobResult(BaseModel):
    file_path: str
    success: bool
    source_id: Optional[uuid.UUID] = None
    error_message: Optional[str] = None

    model_config = ConfigDict(frozen=True)


class AudioExtractionJobResult(BaseModel):
    source_id: uuid.UUID
    success: bool
    extracted_file_id: Optional[uuid.UUID] = None
    error_message: Optional[str] = None

    model_config = ConfigDict(frozen=True)


class VadJobResult(BaseModel):
    """Result of Phase 4: Voice Activity Detection."""
    source_id: uuid.UUID
    success: bool
    vad_map_id: Optional[uuid.UUID] = None
    error_message: Optional[str] = None

    model_config = ConfigDict(frozen=True)

class TranscribeJobResult(BaseModel):
    """Result of Phase 5: Transcription."""
    source_id: uuid.UUID
    success: bool
    transcript_id: Optional[uuid.UUID] = None
    error_message: Optional[str] = None

    model_config = ConfigDict(frozen=True)

# --- Aggregate Batch Results ---

class PhaseBatchResult(BaseModel):
    phase: PipelinePhase
    status: PhaseStatus

    total_processed: int = 0
    successful_count: int = 0
    failed_count: int = 0

    items: List[Any] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True)