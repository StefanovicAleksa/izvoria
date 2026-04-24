# Canvas: clip_extractor_entities_v1
# Path: app/features/clip_extractor/domain/entities.py

import uuid
from pydantic import BaseModel, ConfigDict


class VideoClip(BaseModel):
    """
    Represents the relational link between a Unified Chunk,
    the original Video Source, and the newly extracted physical .mp4 clip.
    """
    id: uuid.UUID
    source_id: uuid.UUID
    chunk_id: uuid.UUID

    # Links to the actual physical .mp4 file stored in the Onyx Vault
    extracted_file_id: uuid.UUID

    start_sec: float
    end_sec: float

    model_config = ConfigDict(from_attributes=True)