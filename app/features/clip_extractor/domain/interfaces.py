# Canvas: clip_extractor_interfaces_v1
# Path: app/features/clip_extractor/domain/interfaces.py

from abc import ABC, abstractmethod
from pathlib import Path
import uuid
from typing import List, Optional

from app.features.clip_extractor.domain.entities import VideoClip

class IFFmpegVideoAdapter(ABC):
    """
    Contract for extracting physical video clips using FFmpeg.
    """
    @abstractmethod
    def extract_clip(self, source_path: str | Path, start_sec: float, end_sec: float) -> Path:
        """
        Extracts a specific time range from a video file.
        Returns the path to the temporary OS file.
        """
        pass

class IVideoClipRepository(ABC):
    """
    Contract for storing and retrieving the VideoClip relationships.
    """
    @abstractmethod
    def create_bulk(self, clips: List[VideoClip]) -> List[VideoClip]:
        """Saves multiple clip records to the database in a single transaction."""
        pass

    @abstractmethod
    def get_by_chunk_id(self, chunk_id: uuid.UUID) -> Optional[VideoClip]:
        """Retrieves the exact video clip corresponding to a specific RAG chunk."""
        pass

    @abstractmethod
    def get_by_source_id(self, source_id: uuid.UUID) -> List[VideoClip]:
        """Retrieves all extracted clips for a given source (used for Idempotency)."""
        pass