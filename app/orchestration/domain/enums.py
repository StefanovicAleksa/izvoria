# Canvas: orchestration_enums_v9
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
    DIARIZE = "diarize"
    HALLUCINATION_SCRUBBER = "hallucination_scrubber" # Phase 6.5
    SENTENCE_RECONSTRUCTION = "sentence_reconstruction"
    INTELLIGENCE_BRIDGE = "intelligence_bridge" # Kept in domain, benched from active pipeline
    DOMAIN_PHONETIC_CORRECTION = "phonetic_correction"
    CONTEXTUAL_RETRIEVAL = "contextual_retrieval" # Phase 9
    VISUAL_CUE_DETECTION = "visual_cue_detection"  # Phase 10
    FRAME_EXTRACTION = "frame_extraction"  # Phase 11
    VISUAL_ANALYSIS = "visual_analysis"  # Phase 12

class PhaseStatus(str, Enum):
    """The execution status of an entire batch phase."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"