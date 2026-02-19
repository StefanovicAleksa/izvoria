# File: app/orchestration/domain/entities.py

import uuid
from typing import List, Optional
from pydantic import BaseModel, ConfigDict


class ScanJobResult(BaseModel):
    """Result of the Directory Scan orchestration step."""
    target_directory: str
    total_found: int
    file_paths: List[str]

    model_config = ConfigDict(frozen=True)


class IngestJobResult(BaseModel):
    """Result of a single File Ingest orchestration step."""
    file_path: str
    success: bool

    # Populated if the ingest was successful
    source_id: Optional[uuid.UUID] = None

    # Populated if the ingest failed
    error_message: Optional[str] = None

    model_config = ConfigDict(frozen=True)