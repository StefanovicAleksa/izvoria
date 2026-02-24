# filename: transcription_interfaces_v1.py
# path: app/features/transcription/domain/interfaces.py
import uuid
from abc import ABC, abstractmethod
from typing import Optional

from app.core.model_manager.domain.interfaces import IModelLoader
from app.core.model_manager.domain.entities import LoadedModel
from app.features.vad.domain.entities import VadMap
from app.features.transcription.domain.entities import Transcript


class ITranscriptionAdapter(IModelLoader, ABC):
    """
    Contract for the AI transcription model (e.g., Faster-Whisper).
    Inherits from IModelLoader to support strict VRAM management via the Orchestrator.
    """

    @abstractmethod
    def transcribe(
        self,
        model: LoadedModel,
        audio_path: str,
        vad_map: VadMap,
        source_id: uuid.UUID
    ) -> Transcript:
        """
        Takes the loaded model, the raw audio file, and the pre-computed VAD Map.
        Should slice the audio according to the VAD Map to skip silence and transcribe
        only the human speech segments.
        """
        pass


class ITranscriptRepository(ABC):
    """
    Contract for saving the completed Transcript to the database.
    Like the VadMap, this will likely be stored as JSONB for high-performance retrieval.
    """

    @abstractmethod
    def save(self, transcript: Transcript) -> Transcript:
        """Saves a new Transcript to the database."""
        pass

    @abstractmethod
    def get_by_source_id(self, source_id: uuid.UUID) -> Optional[Transcript]:
        """Retrieves a previously computed Transcript for a given Source."""
        pass