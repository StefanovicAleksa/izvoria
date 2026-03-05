# Canvas: jobs_enums_v4
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
    VAD = "vad" # voice activity detection
    TRANSCRIBE = "transcribe"
    DIARIZE = "diarize" # speaker diarization
    INTELLIGENCE_BRIDGE = "intelligence_bridge"
    SENTENCE_RECONSTRUCTION = "sentence_reconstruction"