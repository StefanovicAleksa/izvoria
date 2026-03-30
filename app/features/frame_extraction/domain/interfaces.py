# Canvas: frame_extraction_interfaces_v1
# Path: app/features/frame_extraction/domain/interfaces.py

import uuid
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from pathlib import Path

from app.features.contextual_retrieval.domain.entities import ContextualizedChunk
from app.features.frame_extraction.domain.entities import ExtractedChunkFrames


class IFFmpegExtractionAdapter(ABC):
    """Contract for extracting raw frames via OS-level FFmpeg."""

    @abstractmethod
    def extract_raw_frames(
        self,
        video_path: str | Path,
        output_dir: str | Path,
        chunk: ContextualizedChunk,
        fps: int = 1
    ) -> List[Tuple[float, Path]]:
        """
        Uses fast-seeking to extract frames strictly within the chunk's boundaries.
        Returns a list of tuples containing the exact timestamp and the physical Path to the JPEG.
        """
        pass


class IFrameExtractionRepository(ABC):
    """Contract for persisting the ExtractedChunkFrames to PostgreSQL."""

    @abstractmethod
    def save(self, chunk_frames: ExtractedChunkFrames) -> ExtractedChunkFrames:
        pass

    @abstractmethod
    def get_by_chunk_id(self, chunk_id: uuid.UUID) -> Optional[ExtractedChunkFrames]:
        pass