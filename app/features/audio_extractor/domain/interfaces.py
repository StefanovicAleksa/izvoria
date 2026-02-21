# filename: audio_extractor_interfaces_v2.py
from abc import ABC, abstractmethod
from pathlib import Path
import uuid
from typing import Optional

from app.features.audio_extractor.domain.entities import AudioExtraction

class IFFmpegAdapter(ABC):
    """
    Contract for the external FFmpeg tool.
    The implementation will live in the Data layer.
    """

    @abstractmethod
    def extract_audio(self, source_path: str | Path) -> Path:
        """
        Extracts audio from a video file.
        Handles temporary file generation and returns the path to the resulting audio file.
        """
        pass

class IAudioExtractionRepository(ABC):
    """
    Contract for storing extraction linkage history.
    """
    @abstractmethod
    def create(self, extraction: AudioExtraction) -> AudioExtraction:
        """Saves a new extraction record to the database."""
        pass

    @abstractmethod
    def get_latest_by_source_id(self, source_id: uuid.UUID) -> Optional[AudioExtraction]:
        """
        Retrieves the most recent extraction record for a given Video Source.
        Returns None if no extraction has been logged for this source.
        """
        pass