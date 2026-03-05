# Canvas: orchestration_entities_v6
# Path: app/orchestration/domain/entities.py

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


class DiarizeJobResult(BaseModel):
    """Result of Phase 6: Speaker Diarization."""
    source_id: uuid.UUID
    success: bool
    diarization_map_id: Optional[uuid.UUID] = None
    error_message: Optional[str] = None

    model_config = ConfigDict(frozen=True)


class IntelligenceBridgeJobResult(BaseModel):
    """Result of Phase: Intelligence Bridge (Multimodal Scene Mapping)."""
    source_id: uuid.UUID
    success: bool
    scene_map_id: Optional[uuid.UUID] = None
    error_message: Optional[str] = None

    model_config = ConfigDict(frozen=True)


class SentenceReconstructionJobResult(BaseModel):
    """Result of Phase 7: Sentence Reconstruction."""
    source_id: uuid.UUID
    success: bool
    reconstructed_transcript_id: Optional[uuid.UUID] = None
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