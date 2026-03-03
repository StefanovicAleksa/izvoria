# Canvas: orchestration_enums_v3
# Path: app/orchestration/domain/enums.py

from enum import Enum

class PipelinePhase(str, Enum):
    """The sequential phases of the Onyx Media Pipeline."""
    SCAN = "scan"
    INGEST = "ingest"
    AUDIO_EXTRACTION = "audio_extraction"
    VAD = "vad"
    TRANSCRIBE = "transcribe"
    DIARIZE = "diarize"
    INTELLIGENCE_BRIDGE = "intelligence_bridge"

class PhaseStatus(str, Enum):
    """The execution status of an entire batch phase."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"