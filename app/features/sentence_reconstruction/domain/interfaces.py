# Canvas: sentence_interfaces_v3
# Path: app/features/sentence_reconstruction/domain/interfaces.py

import uuid
from abc import ABC, abstractmethod
from typing import List, Optional

from app.core.model_manager.domain.interfaces import IModelLoader
from app.core.model_manager.domain.entities import LoadedModel
from app.features.sentence_reconstruction.domain.entities import ReconstructedTranscript


class ISentenceAdapter(IModelLoader, ABC):
    """Contract for the SaT neural segmentation engine."""

    @abstractmethod
    def segment_text(self, model: LoadedModel, text: str) -> List[str]:
        pass


class ISentenceRepository(ABC):
    """Contract for persisting the reconstructed transcript to PostgreSQL."""

    @abstractmethod
    def save(self, transcript: ReconstructedTranscript) -> ReconstructedTranscript: # FIXED: Naming convention
        pass

    @abstractmethod
    def get_by_source_id(self, source_id: uuid.UUID) -> Optional[ReconstructedTranscript]:
        pass