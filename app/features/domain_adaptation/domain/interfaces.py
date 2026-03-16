# Canvas: domain_adaptation_interfaces_v2
# Path: app/features/domain_adaptation/domain/interfaces.py

import uuid
from abc import ABC, abstractmethod
from typing import List, Optional

from app.core.model_manager.domain.interfaces import IModelLoader
from app.core.model_manager.domain.entities import LoadedModel
from app.features.domain_adaptation.domain.entities import ScoutResult, DomainGlossary, AudioSample

class IScoutWhisperAdapter(IModelLoader, ABC):
    """
    Contract for the GPU-bound faster-whisper-tiny.en Scout model.
    Inherits from IModelLoader to support Orchestrator VRAM allocation.
    """
    @abstractmethod
    def transcribe_samples(
        self,
        model: LoadedModel,
        samples: List[AudioSample],
        source_id: uuid.UUID
    ) -> ScoutResult:
        pass

class IClassifierAdapter(IModelLoader, ABC):
    """
    Contract for the GPU-bound zero-shot classifier (e.g., DeBERTa).
    """
    @abstractmethod
    def classify_domain(
        self,
        model: LoadedModel,
        text: str,
        candidate_labels: List[str]
    ) -> str:
        pass

class IDomainAdaptationRepository(ABC):
    """Contract for persisting the final Domain Glossary to PostgreSQL."""
    @abstractmethod
    def save_glossary(self, glossary: DomainGlossary) -> DomainGlossary:
        pass

    @abstractmethod
    def get_glossary_by_source_id(self, source_id: uuid.UUID) -> Optional[DomainGlossary]:
        pass