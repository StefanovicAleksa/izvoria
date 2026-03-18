# Canvas: phonetic_interfaces_v1
# Path: app/features/phonetic_correction/domain/interfaces.py

import uuid
from abc import ABC, abstractmethod
from typing import List, Optional

from app.core.model_manager.domain.interfaces import IModelLoader
from app.core.model_manager.domain.entities import LoadedModel

from app.features.phonetic_correction.domain.entities import (
    PhoneticPatch,
    PhoneticCorrectedTranscript
)


class IPhoneticPatchAdapter(IModelLoader, ABC):
    """
    Contract for the AI Model Adapter (e.g., Qwen 2.5).
    Enforces the 'Read-Only JSON Patch' strategy.
    """

    @abstractmethod
    def generate_patches(
        self,
        model: LoadedModel,
        text_chunk: str,
        domain_dictionary: str
    ) -> List[PhoneticPatch]:
        """
        Scans a chunk of text against a static domain dictionary.
        Returns a list of strict PhoneticPatch objects. Does NOT return rewritten text.
        """
        pass


class IPhoneticRepository(ABC):
    """
    Contract for persisting the finalized transcript and its audit trail.
    """

    @abstractmethod
    def save(self, transcript: PhoneticCorrectedTranscript) -> PhoneticCorrectedTranscript:
        pass

    @abstractmethod
    def get_by_source_id(self, source_id: uuid.UUID) -> Optional[PhoneticCorrectedTranscript]:
        pass