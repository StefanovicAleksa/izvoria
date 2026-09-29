# filename: app/features/storage/domain/entities.py
import uuid
from enum import Enum
from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


class SourceType(str, Enum):
    VIDEO = "video"
    AUDIO = "audio"
    TEXT = "text"
    DOCUMENT = "document"
    IMAGE = "image"


class File(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    file_path: str
    file_name: str
    file_size_bytes: int
    mime_type: str | None = None
    file_metadata: Dict[str, Any] = Field(default_factory=dict)
    is_processed: bool = False
    created_at: datetime = Field(default_factory=datetime.now)

    model_config = ConfigDict(from_attributes=True)


class Source(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    name: str
    source_type: SourceType
    content_hash: str
    # The language the file was ingested in ("sr"); None while nothing binds it (see SourceModel.language)
    language: Optional[str] = None

    # The Foreign Key linking to the independent physical file
    file_id: uuid.UUID

    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    # For performance/convenience, holding the loaded file data
    file: Optional[File] = None

    model_config = ConfigDict(from_attributes=True)