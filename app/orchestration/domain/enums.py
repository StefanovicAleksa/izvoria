# filename: orchestration_enums_v1.py
from enum import Enum

class PipelinePhase(str, Enum):
    """The sequential phases of the Onyx Media Pipeline."""
    SCAN = "scan"
    INGEST = "ingest"
    AUDIO_EXTRACTION = "audio_extraction"
    TRANSCRIBE = "transcribe"
    VISION = "vision"

class PhaseStatus(str, Enum):
    """The execution status of an entire batch phase."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL_SUCCESS = "partial_success"  # E.g., 90/100 files extracted audio successfully
    FAILED = "failed"