# Canvas: transcription_interfaces_v6
# Path: app/features/transcription/domain/interfaces.py

import uuid
from abc import ABC, abstractmethod
from typing import List, Optional

from app.core.model_manager.domain.entities import LoadedModel
from app.core.model_manager.domain.interfaces import IModelLoader
from app.features.vad.domain.entities import VadMap
from app.features.transcription.domain.entities import Transcript, TranscriptSegment

class ITranscriptionAdapter(IModelLoader, ABC):
    """Contract for the heavy GPU-bound transcription model."""

    @abstractmethod
    def transcribe(
            self,
            model: LoadedModel,
            audio_path: str,
            vad_map: VadMap,
            language: str,
            hotwords: Optional[str] = None,
            initial_prompt: Optional[str] = None
    ) -> List[TranscriptSegment]:
        pass

class ITranscriptRepository(ABC):
    """Contract for persisting the Transcript to PostgreSQL."""

    @abstractmethod
    def save(self, transcript: Transcript) -> Transcript:
        pass

    @abstractmethod
    def get_by_source_id(self, source_id: uuid.UUID) -> Optional[Transcript]:
        pass

    @abstractmethod
    def update(self, transcript: Transcript) -> Transcript:
        """Updates an existing Transcript in the database."""
        pass