# Canvas: frame_extraction_entities_v1
# Path: app/features/frame_extraction/domain/entities.py

import uuid
from typing import List
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict


class ExtractedFrame(BaseModel):
    """
    Represents a high-quality, deduplicated frame vaulted in the Storage feature.
    """
    frame_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    file_id: uuid.UUID = Field(description="The UUID of the physical JPEG in the Storage Vault.")
    timestamp_sec: float = Field(description="The exact second this frame was captured.")
    phash_value: str = Field(description="The 64-bit perceptual hash used for deduplication.")
    quality_score: float = Field(description="The sharpness/clarity score (e.g., Variance of Laplacian).")

    model_config = ConfigDict(frozen=True)


class ExtractedChunkFrames(BaseModel):
    """
    The Aggregate Root for the Frame Extraction phase.
    Maps a ContextualizedChunk to its finalized, high-quality visual frames.
    """
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    source_id: uuid.UUID = Field(description="The UUID of the original Video Source.")
    chunk_id: uuid.UUID = Field(description="The UUID of the target ContextualizedChunk.")

    frames: List[ExtractedFrame] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(from_attributes=True)