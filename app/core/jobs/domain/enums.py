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