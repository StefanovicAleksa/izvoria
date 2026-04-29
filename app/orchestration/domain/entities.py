# Canvas: orchestration_entities_v23
# Path: app/orchestration/domain/entities.py

import uuid
from typing import List, Optional, Any
from pydantic import BaseModel, Field, ConfigDict
from app.orchestration.domain.enums import PipelinePhase, PhaseStatus

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
    source_id: uuid.UUID
    success: bool
    vad_map_id: Optional[uuid.UUID] = None
    error_message: Optional[str] = None
    model_config = ConfigDict(frozen=True)

class DomainAdaptationJobResult(BaseModel):
    source_id: uuid.UUID
    success: bool
    glossary_id: Optional[uuid.UUID] = None
    error_message: Optional[str] = None
    model_config = ConfigDict(frozen=True)

class TranscribeJobResult(BaseModel):
    source_id: uuid.UUID
    success: bool
    transcript_id: Optional[uuid.UUID] = None
    error_message: Optional[str] = None
    model_config = ConfigDict(frozen=True)

class HallucinationCorrectionJobResult(BaseModel):
    source_id: uuid.UUID
    success: bool
    transcript_id: Optional[uuid.UUID] = None
    error_message: Optional[str] = None
    model_config = ConfigDict(frozen=True)

class DiarizeJobResult(BaseModel):
    source_id: uuid.UUID
    success: bool
    diarization_map_id: Optional[uuid.UUID] = None
    error_message: Optional[str] = None
    model_config = ConfigDict(frozen=True)

class HallucinationScrubberJobResult(BaseModel):
    source_id: uuid.UUID
    success: bool
    scrubbed_transcript_id: Optional[uuid.UUID] = None
    error_message: Optional[str] = None
    model_config = ConfigDict(frozen=True)

class IntelligenceBridgeJobResult(BaseModel):
    source_id: uuid.UUID
    success: bool
    scene_map_id: Optional[uuid.UUID] = None
    error_message: Optional[str] = None
    model_config = ConfigDict(frozen=True)

class SentenceReconstructionJobResult(BaseModel):
    source_id: uuid.UUID
    success: bool
    reconstructed_transcript_id: Optional[uuid.UUID] = None
    error_message: Optional[str] = None
    model_config = ConfigDict(frozen=True)

class DomainPhoneticCorrectionJobResult(BaseModel):
    source_id: uuid.UUID
    success: bool
    domain_phonetic_transcript_id: Optional[uuid.UUID] = None
    error_message: Optional[str] = None
    model_config = ConfigDict(frozen=True)

class ContextualRetrievalJobResult(BaseModel):
    source_id: uuid.UUID
    success: bool
    contextual_transcript_id: Optional[uuid.UUID] = None
    error_message: Optional[str] = None
    model_config = ConfigDict(frozen=True)

class VisualCueDetectionJobResult(BaseModel):
    source_id: uuid.UUID
    success: bool
    evaluated_transcript_id: Optional[uuid.UUID] = None
    error_message: Optional[str] = None
    model_config = ConfigDict(frozen=True)

class FrameExtractionJobResult(BaseModel):
    source_id: uuid.UUID
    success: bool
    extracted_chunks_count: int = 0
    error_message: Optional[str] = None
    model_config = ConfigDict(frozen=True)

class VisualAnalysisJobResult(BaseModel):
    source_id: uuid.UUID
    success: bool
    analyzed_chunks_count: int = 0
    error_message: Optional[str] = None
    model_config = ConfigDict(frozen=True)

class VisualHallucinationCorrectionJobResult(BaseModel):
    source_id: uuid.UUID
    success: bool
    evaluated_transcript_id: Optional[uuid.UUID] = None
    error_message: Optional[str] = None
    model_config = ConfigDict(frozen=True)

class VisualContextSynthesisJobResult(BaseModel):
    source_id: uuid.UUID
    success: bool
    visual_timeline_id: Optional[uuid.UUID] = None
    error_message: Optional[str] = None
    model_config = ConfigDict(frozen=True)

class ContextUnificationJobResult(BaseModel):
    source_id: uuid.UUID
    success: bool
    unified_transcript_id: Optional[uuid.UUID] = None
    error_message: Optional[str] = None
    model_config = ConfigDict(frozen=True)

class ClipExtractionJobResult(BaseModel):
    source_id: uuid.UUID
    success: bool
    extracted_clips_count: int = 0
    error_message: Optional[str] = None
    model_config = ConfigDict(frozen=True)

class RagIndexingJobResult(BaseModel):
    source_id: uuid.UUID
    success: bool
    dense_vectors_saved: int = 0
    sparse_vectors_saved: int = 0
    error_message: Optional[str] = None
    model_config = ConfigDict(frozen=True)

class RagRetrievalJobResult(BaseModel):
    search_id: uuid.UUID
    success: bool
    retrieved_chunks_count: int = 0
    error_message: Optional[str] = None
    model_config = ConfigDict(frozen=True)

class RetrievalScrubbingJobResult(BaseModel):
    search_id: uuid.UUID
    success: bool
    surviving_chunks_count: int = 0
    error_message: Optional[str] = None
    model_config = ConfigDict(frozen=True)

class PhaseBatchResult(BaseModel):
    phase: PipelinePhase
    status: PhaseStatus
    total_processed: int = 0
    successful_count: int = 0
    failed_count: int = 0
    items: List[Any] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    model_config = ConfigDict(frozen=True)