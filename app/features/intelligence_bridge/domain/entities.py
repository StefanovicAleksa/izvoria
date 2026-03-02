import uuid
from typing import List, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict


class EnrichedScene(BaseModel):
    """A definitive, deduplicated keyframe combining visual and semantic data."""
    timestamp_sec: float
    extracted_file_id: uuid.UUID
    phash_value: Optional[str] = None

    extraction_method: str = Field(
        description="Indicates if this was caught by 'global_pass' or 'semantic_pass'"
    )
    semantic_context: Optional[str] = Field(
        default=None,
        description="The specific Whisper transcript text that triggered this frame, if applicable."
    )

    model_config = ConfigDict(frozen=True)


class IntelligentSceneMap(BaseModel):
    """The ultimate multimodal representation of a video."""
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    source_id: uuid.UUID = Field(description="The UUID of the original Video Source.")

    scenes: List[EnrichedScene] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(from_attributes=True)