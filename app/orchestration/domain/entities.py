# Canvas: orchestration_entities_v24
# Path: app/orchestration/domain/entities.py

import uuid
from typing import List, Optional, Any
from pydantic import BaseModel, Field, ConfigDict
from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.features.qa_generation.domain.entities import QAResponse

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

class DocumentConversionJobResult(BaseModel):
    source_id: uuid.UUID
    success: bool
    converted_document_id: Optional[uuid.UUID] = None
    pages_total: int = 0
    pages_ocr: int = 0
    # Pages whose OCR lost detected lines, and scanned pages with no text at all: indexed incompletely or not
    # at all, and reported rather than hidden
    pages_ocr_incomplete: int = 0
    pages_unreadable: int = 0
    error_message: Optional[str] = None
    model_config = ConfigDict(frozen=True)

class ContextualRetrievalJobResult(BaseModel):
    source_id: uuid.UUID
    success: bool
    contextual_transcript_id: Optional[uuid.UUID] = None
    error_message: Optional[str] = None
    model_config = ConfigDict(frozen=True)

class ImageDescriptionJobResult(ContextualRetrievalJobResult):
    """
    An image described and turned into its one-chunk contextualized transcript. A ContextualRetrievalJobResult, so
    unification and indexing take it unchanged; the counts are what the ingestion report says about its text.
    """
    # OCR ran on the image; False on an appliance without OCR, whose text is then unverified
    ocr_checked: bool = False
    # Text items the vision model read that OCR did not: not indexed
    text_items_dropped: int = 0
    # OCR lost lines it had detected
    ocr_incomplete: bool = False
    # Low-confidence OCR lines the vision model did not confirm: not indexed
    ocr_lines_withheld: int = 0


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

class StorageCompactionJobResult(BaseModel):
    source_id: uuid.UUID
    success: bool
    # The WAV dropped plus what replacing the original with its compact copy saved
    bytes_saved: int = 0
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

class RagRerankingJobResult(BaseModel):
    search_id: uuid.UUID
    success: bool
    reranked_chunks_count: int = 0
    error_message: Optional[str] = None
    model_config = ConfigDict(frozen=True)

class RetrievalScrubbingJobResult(BaseModel):
    search_id: uuid.UUID
    success: bool
    surviving_chunks_count: int = 0
    error_message: Optional[str] = None
    model_config = ConfigDict(frozen=True)

# NEW: State tracker for the Chat Generation
class QAGenerationJobResult(BaseModel):
    search_id: uuid.UUID
    success: bool
    generated_answer_preview: Optional[str] = None
    citations_count: int = 0
    error_message: Optional[str] = None
    model_config = ConfigDict(frozen=True)

# NEW: State tracker for one Reasoner turn
class ReasonerJobResult(BaseModel):
    search_id: uuid.UUID
    success: bool
    decision: Optional[str] = None
    error_message: Optional[str] = None
    model_config = ConfigDict(frozen=True)

# NEW: Reasoned Answer pipeline result — the engine returns this directly as chat_reasoned()'s
# response type (mirrors how OnyxEngine.chat() already returns QAResponse, a feature entity,
# unchanged; this lives in orchestration rather than app/engine/domain/entities.py so the
# pipeline that builds it doesn't have to depend upward on the engine layer).
class ReasoningIterationTrace(BaseModel):
    iteration: int
    decision: str
    interim_analysis: str
    refined_query: Optional[str] = None
    chunks_added: int = 0
    model_config = ConfigDict(frozen=True)

class ReasonedChatResponse(BaseModel):
    qa_response: Optional[QAResponse] = None
    clarification_question: Optional[str] = None
    iteration_trace: List[ReasoningIterationTrace] = Field(default_factory=list)
    critique_unresolved: bool = False
    total_iterations: int = 0
    # The first search query (a follow-up rewritten to stand alone) and every chunk the evidence pool held,
    # best first, so the conversation can show them to later turns
    standalone_query: Optional[str] = None
    evidence_chunk_ids: List[uuid.UUID] = Field(default_factory=list)
    model_config = ConfigDict(frozen=True)

# One answered turn of a multi-turn chat: the answer, the query it searched with, and the chunks it showed first
class ConversationChatResult(BaseModel):
    qa_response: QAResponse
    standalone_query: str
    new_chunk_ids: List[uuid.UUID] = Field(default_factory=list)
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


# ==========================================
# Deep Research Report mode
# ==========================================

class ResearchScopingJobResult(BaseModel):
    report_id: Optional[uuid.UUID] = None
    success: bool = False
    section_count: int = 0
    sub_question_count: int = 0
    error_message: Optional[str] = None
    model_config = ConfigDict(frozen=True)


class EvidenceSweepJobResult(BaseModel):
    """One search of the gathering stage -- a sub-question, or the per-document sweep as a whole."""
    report_id: uuid.UUID
    success: bool
    query: str
    chunks_found: int = 0
    is_sweep: bool = False
    swept_source_count: int = 0
    error_message: Optional[str] = None
    model_config = ConfigDict(frozen=True)


class ReportOutlineJobResult(BaseModel):
    report_id: uuid.UUID
    success: bool
    # How many of the approved sections the gathered evidence could actually support. The gap between
    # this and section_count is the report's honest coverage, and is surfaced rather than smoothed over.
    covered_section_count: int = 0
    section_count: int = 0
    error_message: Optional[str] = None
    model_config = ConfigDict(frozen=True)


class ReportSectionJobResult(BaseModel):
    report_id: uuid.UUID
    section_key: str
    success: bool
    status: Optional[str] = None
    cited_count: int = 0
    error_message: Optional[str] = None
    model_config = ConfigDict(frozen=True)
