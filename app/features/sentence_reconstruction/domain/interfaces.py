# Canvas: sentence_interfaces_v4
# Path: app/features/sentence_reconstruction/domain/interfaces.py

import uuid
from abc import ABC, abstractmethod
from typing import List, Optional

from app.core.model_manager.domain.interfaces import IModelLoader
from app.core.model_manager.domain.entities import LoadedModel

from app.features.transcription.domain.entities import TranscriptWord
from app.features.sentence_reconstruction.domain.entities import PunctuatedWord, ReconstructedTranscript


class IPunctuationAdapter(IModelLoader, ABC):
    """
    Contract for the Token Classification engine (e.g., oliverguhr/fullstop).
    Replaces the old ISentenceAdapter (SaT model).
    """

    @property
    def requires_model(self) -> bool:
        """False for an adapter that tags words without loading anything; callers then skip acquiring a model."""
        return True

    @abstractmethod
    def predict_punctuation(self, model: LoadedModel, words: List[TranscriptWord]) -> List[PunctuatedWord]:
        """
        Takes a continuous stream of acoustic words and tags each one with a punctuation marker.
        Does NOT alter the text, drop characters, or hallucinate.
        Returns an exact 1:1 mapped list.
        """
        pass


class ISentenceRepository(ABC):
    """Contract for persisting the reconstructed transcript to PostgreSQL."""

    @abstractmethod
    def save(self, transcript: ReconstructedTranscript) -> ReconstructedTranscript:
        pass

    @abstractmethod
    def get_by_source_id(self, source_id: uuid.UUID) -> Optional[ReconstructedTranscript]:
        pass