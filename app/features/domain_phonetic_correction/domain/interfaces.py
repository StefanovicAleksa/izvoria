# Path: app/features/domain_phonetic_correction/domain/interfaces.py

import uuid
from abc import ABC, abstractmethod
from typing import List, Optional

from app.core.model_manager.domain.interfaces import IModelLoader
from app.core.model_manager.domain.entities import LoadedModel

from app.features.domain_phonetic_correction.domain.entities import (
    DomainPhoneticCorrectedTranscript,
    PhoneticCandidate
)

class IDomainPhoneticPatchAdapter(IModelLoader, ABC):
    @abstractmethod
    def verify_candidates(
        self,
        model: LoadedModel,
        sentence_text: str,
        previous_text: str,
        candidates: List[PhoneticCandidate],
        domain: str
    ) -> List[bool]:
        """
        Judges, for each candidate, whether the speaker said the dictionary term and the transcript mis-heard it.
        Returns exactly one verdict per candidate, in order; anything unusable counts as False (not a mis-hearing).
        """
        pass

class IDomainPhoneticRepository(ABC):
    @abstractmethod
    def save(self, transcript: DomainPhoneticCorrectedTranscript) -> DomainPhoneticCorrectedTranscript:
        pass

    @abstractmethod
    def get_by_source_id(self, source_id: uuid.UUID) -> Optional[DomainPhoneticCorrectedTranscript]:
        pass
