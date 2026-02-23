# File: app/features/vad/domain/interfaces.py
import uuid
from abc import ABC, abstractmethod
from typing import Optional

from app.core.model_manager.domain.interfaces import IModelLoader
from app.core.model_manager.domain.entities import LoadedModel
from app.features.vad.domain.entities import VadMap


class IVadModelAdapter(IModelLoader, ABC):
    """
    Contract for the actual AI model (e.g., Silero or NeMo).
    It inherits from IModelLoader so the Orchestrator can load/unload it,
    but it adds the generate_vad_map() method specific to VAD logic.
    """

    @abstractmethod
    def generate_vad_map(self, model: LoadedModel, audio_path: str, source_id: uuid.UUID) -> VadMap:
        """
        Analyzes the full audio file and generates the virtual map of human speech.
        """
        pass


class IVadRepository(ABC):
    """
    Contract for saving the VAD Map to Postgres.
    By storing the map as a JSONB column in the database, we ensure
    crash-resilience without cluttering the physical Storage Vault.
    """

    @abstractmethod
    def save_map(self, vad_map: VadMap) -> VadMap:
        """Saves a new VAD Map to the database."""
        pass

    @abstractmethod
    def get_map_by_source_id(self, source_id: uuid.UUID) -> Optional[VadMap]:
        """Retrieves a previously computed VAD Map for a given Source."""
        pass