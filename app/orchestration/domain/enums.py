# Canvas: orchestration_enums_v7
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
    SENTENCE_RECONSTRUCTION = "sentence_reconstruction"
    INTELLIGENCE_BRIDGE = "intelligence_bridge" # Kept in domain, benched from active pipeline
    PHONETIC_CORRECTION = "phonetic_correction"

class PhaseStatus(str, Enum):
    """The execution status of an entire batch phase."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"