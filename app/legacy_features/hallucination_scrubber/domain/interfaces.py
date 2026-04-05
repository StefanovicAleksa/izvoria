# Canvas: hallucination_scrubber_interfaces_v1
# Path: app/features/hallucination_scrubber/domain/interfaces.py

import uuid
from abc import ABC, abstractmethod
from typing import List, Dict, Optional

from app.core.model_manager.domain.interfaces import IModelLoader
from app.core.model_manager.domain.entities import LoadedModel
from app.legacy_features.hallucination_scrubber.domain.entities import HallucinationScrubbedTranscript


class IZeroShotAdapter(IModelLoader, ABC):
    """
    Contract for the Zero-Shot Classification model (DeBERTa).
    Inherits from IModelLoader to strictly integrate with the VRAM Orchestrator.
    """

    @abstractmethod
    def classify(self, model: LoadedModel, text: str, candidate_labels: List[str]) -> Dict[str, float]:
        """
        Evaluates a single text string against candidate labels.
        Returns a dictionary mapping each label to its probability score (0.0 to 1.0).
        """
        pass

    @abstractmethod
    def classify_batch(self, model: LoadedModel, texts: List[str], candidate_labels: List[str]) -> List[
        Dict[str, float]]:
        """
        Batch processing for sub-clauses to maximize GPU throughput.
        Returns a list of dictionaries, preserving the index order of the input texts.
        """
        pass


class IHallucinationRepository(ABC):
    """
    Contract for persisting the Scrubbed Transcript and its Audit Trail to PostgreSQL.
    """

    @abstractmethod
    def save(self, transcript: HallucinationScrubbedTranscript) -> HallucinationScrubbedTranscript:
        pass

    @abstractmethod
    def get_by_source_id(self, source_id: uuid.UUID) -> Optional[HallucinationScrubbedTranscript]:
        pass