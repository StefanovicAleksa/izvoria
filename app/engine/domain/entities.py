# Canvas: engine_entities_v1
# Path: app/engine/domain/entities.py

from typing import Dict
from pydantic import BaseModel, Field, ConfigDict


class IngestionReport(BaseModel):
    """The structured result of pointing the Onyx Engine at a directory."""
    target_path: str
    total_files_scanned: int = 0
    successful_ingestions: int = 0
    failed_ingestions: int = 0

    # Maps file paths to error messages (e.g., {"/vault/video.mp4": "CUDA OOM"})
    errors: Dict[str, str] = Field(default_factory=dict)

    # Telemetry
    total_duration_seconds: float = 0.0
    vectors_created: int = 0
    clips_extracted: int = 0

    model_config = ConfigDict(frozen=True)