# Canvas: domain_phonetic_interfaces_v2
# Path: app/features/domain_phonetic_correction/domain/interfaces.py

import uuid
from abc import ABC, abstractmethod
from typing import List, Optional

from app.core.model_manager.domain.interfaces import IModelLoader
from app.core.model_manager.domain.entities import LoadedModel

from app.features.domain_phonetic_correction.domain.entities import (
    DomainPhoneticPatch,
    DomainPhoneticCorrectedTranscript
)

class IDomainPhoneticPatchAdapter(IModelLoader, ABC):
    @abstractmethod
    def generate_patches(
        self,
        model: LoadedModel,
        text_chunk: str,
        domain_dictionary: str
    ) -> List[DomainPhoneticPatch]:
        pass

class IDomainPhoneticRepository(ABC):
    @abstractmethod
    def save(self, transcript: DomainPhoneticCorrectedTranscript) -> DomainPhoneticCorrectedTranscript:
        pass

    @abstractmethod
    def get_by_source_id(self, source_id: uuid.UUID) -> Optional[DomainPhoneticCorrectedTranscript]:
        pass