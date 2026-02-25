# Canvas: diarization_interfaces_v1
# Path: app/features/diarization/domain/interfaces.py

import uuid
from abc import ABC, abstractmethod
from typing import Optional

from app.core.model_manager.domain.interfaces import IModelLoader
from app.core.model_manager.domain.entities import LoadedModel
from app.features.vad.domain.entities import VadMap
from app.features.diarization.domain.entities import DiarizationMap


class IDiarizationModelAdapter(IModelLoader, ABC):
    """
    Contract for the AI diarization model (e.g., NeMo Speaker Diarization).
    Inherits from IModelLoader to support strict VRAM management via the Orchestrator.
    """

    @abstractmethod
    def generate_diarization_map(
        self,
        model: LoadedModel,
        audio_path: str,
        vad_map: VadMap,
        source_id: uuid.UUID
    ) -> DiarizationMap:
        """
        Takes the loaded model, raw audio, and pre-computed VAD Map.
        Clustering algorithms rely on the VAD map to know exactly where the speech is,
        ignoring dead air to save compute and improve accuracy.
        """
        pass


class IDiarizationRepository(ABC):
    """
    Contract for saving the completed DiarizationMap to the database.
    Like the VadMap, this will be stored as JSONB for crash-resilience and quick retrieval.
    """

    @abstractmethod
    def save_map(self, diarization_map: DiarizationMap) -> DiarizationMap:
        """Saves a new Diarization Map to the database."""
        pass

    @abstractmethod
    def get_map_by_source_id(self, source_id: uuid.UUID) -> Optional[DiarizationMap]:
        """Retrieves a previously computed Diarization Map for a given Source."""
        pass