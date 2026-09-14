# Canvas: jobs_enums_v16
# Path: app/core/jobs/domain/enums.py

from enum import Enum


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "canceled"


class JobType(str, Enum):
    DIRECTORY_SCAN = "directory_scan"
    FILE_INGEST = "file_ingest"
    AUDIO_EXTRACTION = "audio_extraction"
    VAD = "vad"  # voice activity detection
    DOMAIN_ADAPTATION = "domain_adaptation"
    TRANSCRIBE = "transcribe"
    HALLUCINATION_CORRECTION = "hallucination_correction"  # Phase 5.5
    DIARIZE = "diarize"  # speaker diarization
    HALLUCINATION_SCRUBBER = "hallucination_scrubber"  # Legacy Phase 6.5
    INTELLIGENCE_BRIDGE = "intelligence_bridge"  # Legacy
    SENTENCE_RECONSTRUCTION = "sentence_reconstruction"
    DOMAIN_PHONETIC_CORRECTION = "phonetic_correction"
    CONTEXTUAL_RETRIEVAL = "contextual_retrieval"  # Phase 9
    VISUAL_CUE_DETECTION = "visual_cue_detection"  # Phase 10
    FRAME_EXTRACTION = "frame_extraction"  # Phase 11
    VISUAL_ANALYSIS = "visual_analysis"  # Phase 12
    VISUAL_HALLUCINATION_CORRECTION = "visual_hallucination_correction"  # Phase 13
    VISUAL_CONTEXT_SYNTHESIS = "visual_context_synthesis"  # Phase 14
    CONTEXT_UNIFICATION = "context_unification"  # Phase 15
    CLIP_EXTRACTION = "clip_extraction"  # Phase 15.5
    RAG_INDEXING = "rag_indexing"  # Phase 16

    # NEW: Added both Live Inference Job Types
    RAG_RETRIEVAL = "rag_retrieval"
    RAG_RERANKING = "rag_reranking"
    RETRIEVAL_SCRUBBING = "retrieval_scrubbing"
    QA_GENERATION = "qa_generation"
