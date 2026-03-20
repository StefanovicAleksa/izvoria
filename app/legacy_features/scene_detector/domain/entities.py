# Canvas: scene_detector_entities_v1
# Path: app/features/scene_detector/domain/entities.py

import uuid
from typing import List, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict


class TimeWindow(BaseModel):
    """
    Represents a specific temporal slice of a video.
    Used to constrain CPU-heavy FFmpeg operations to highly relevant semantic moments.
    """
    start_sec: float = Field(ge=0.0, description="Start time in seconds.")
    end_sec: float = Field(ge=0.0, description="End time in seconds.")

    model_config = ConfigDict(frozen=True)

    def contains(self, timestamp: float) -> bool:
        """Helper to check if a specific timestamp falls within this window."""
        return self.start_sec <= timestamp <= self.end_sec


class DetectedScene(BaseModel):
    """
    Represents a single, highly unique visual frame extracted from a video.
    """
    timestamp_sec: float = Field(description="The exact moment in the video the scene occurred.")

    # We map to the generic File ID in our Storage Vault, keeping this domain pure
    extracted_file_id: uuid.UUID = Field(description="The UUID of the physical JPEG in the Storage feature.")

    # Storing the perceptual hash allows us to audit why a frame was kept or dropped later
    phash_value: Optional[str] = Field(default=None, description="The 64-bit perceptual hash of the image.")

    model_config = ConfigDict(frozen=True)


class SceneMap(BaseModel):
    """
    The complete collection of unique visual scenes for a given media source.
    This acts as the master timeline that the Vision Model (Qwen-VL) will iterate over.
    """
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    source_id: uuid.UUID = Field(description="The UUID of the original Video Source.")

    scenes: List[DetectedScene] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(from_attributes=True)

    @property
    def total_scenes(self) -> int:
        return len(self.scenes)