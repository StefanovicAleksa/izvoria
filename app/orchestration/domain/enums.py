# Canvas: orchestration_enums_v11
# Path: app/orchestration/domain/enums.py

from enum import Enum

class PipelinePhase(str, Enum):
    """The sequential phases of the Onyx Media Pipeline."""
    SCAN = "scan"
    INGEST = "ingest"
    AUDIO_EXTRACTION = "audio_extraction"
    VAD = "vad"
    DOMAIN_ADAPTATION = "domain_adaptation"
    TRANSCRIBE = "transcribe"
    HALLUCINATION_CORRECTION = "hallucination_correction" # Phase 5.5
    DIARIZE = "diarize"
    HALLUCINATION_SCRUBBER = "hallucination_scrubber" # Legacy Phase 6.5
    SENTENCE_RECONSTRUCTION = "sentence_reconstruction"
    INTELLIGENCE_BRIDGE = "intelligence_bridge" # Kept in domain, benched from active pipeline
    DOMAIN_PHONETIC_CORRECTION = "phonetic_correction"
    CONTEXTUAL_RETRIEVAL = "contextual_retrieval" # Phase 9
    VISUAL_CUE_DETECTION = "visual_cue_detection"  # Phase 10
    FRAME_EXTRACTION = "frame_extraction"  # Phase 11
    VISUAL_ANALYSIS = "visual_analysis"  # Phase 12
    VISUAL_HALLUCINATION_CORRECTION = "visual_hallucination_correction" # Phase 13
    VISUAL_CONTEXT_SYNTHESIS = "visual_context_synthesis" # Phase 14
    CONTEXT_UNIFICATION = "context_unification"  # Phase 15
    CLIP_EXTRACTION = "clip_extraction"  # Phase 15.5
    RAG_INDEXING = "rag_indexing"  # Phase 16

    # Document ingestion (PDF / DOCX)
    DOCUMENT_CONVERSION = "document_conversion"
    DOCUMENT_CONTEXTUALIZATION = "document_contextualization"

    # NEW: Live Inference Retrieval Phases
    RAG_RETRIEVAL = "rag_retrieval"
    RAG_RERANKING = "rag_reranking"
    RETRIEVAL_SCRUBBING = "retrieval_scrubbing"
    QA_GENERATION = "qa_generation"
    REASONING_TURN = "reasoning_turn"

    # Deep Research Report mode
    RESEARCH_SCOPING = "research_scoping"
    RESEARCH_SWEEP = "research_sweep"
    REPORT_OUTLINE = "report_outline"
    REPORT_SECTION = "report_section"

class PhaseStatus(str, Enum):
    """The execution status of an entire batch phase."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"