# Canvas: domain_adaptation_interfaces_v4
# Path: app/features/domain_adaptation/domain/interfaces.py

import uuid
from abc import ABC, abstractmethod
from typing import List, Optional

from app.core.model_manager.domain.interfaces import IModelLoader
from app.core.model_manager.domain.entities import LoadedModel
from app.features.domain_adaptation.domain.entities import (
    ScoutResult,
    DomainGlossary,
    TimeSlice
)

class IScoutWhisperAdapter(IModelLoader, ABC):
    """Contract for the GPU-bound faster-whisper-small Scout model."""
    @abstractmethod
    def transcribe_samples(
        self,
        model: LoadedModel,
        audio_path: str,
        slices: List[TimeSlice],
        source_id: uuid.UUID
    ) -> ScoutResult:
        """Transcribes the dense audio RAM chunks into a rough preview text."""
        pass

class ILlamaAdapter(IModelLoader, ABC):
    """Contract for the GPU-bound LLM Router (Llama 3.2 3B)."""
    @abstractmethod
    def predict_subdomains(
        self,
        model: LoadedModel,
        scout_text: str,
        candidate_labels: List[str]
    ) -> List[str]:
        """Analyzes the dirty transcript and returns a list of relevant subdomains."""
        pass

class IDomainAdaptationRepository(ABC):
    """Contract for persisting the final Domain Glossary to PostgreSQL."""
    @abstractmethod
    def save_glossary(self, glossary: DomainGlossary) -> DomainGlossary:
        pass

    @abstractmethod
    def get_glossary_by_source_id(self, source_id: uuid.UUID) -> Optional[DomainGlossary]:
        pass