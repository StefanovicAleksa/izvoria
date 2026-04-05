# Canvas: visual_hallucination_interfaces_v2
# Path: app/features/visual_hallucination_correction/domain/interfaces.py

import uuid
from abc import ABC, abstractmethod
from typing import Optional

from app.core.model_manager.domain.interfaces import IModelLoader
from app.core.model_manager.domain.entities import LoadedModel
from app.features.visual_hallucination_correction.domain.entities import (
    VisualHallucinationAudit,
    VisualHallucinationEvaluatedTranscript
)


class IVisualHallucinationAdapter(IModelLoader, ABC):
    """
    Contract for the SLM (e.g., Qwen 2.5 3B) that acts strictly as the hallucination gatekeeper.
    """

    @abstractmethod
    def evaluate_visual_match(
            self,
            model: LoadedModel,
            chunk_id: uuid.UUID,
            text_chunk: str,
            visual_description: str
    ) -> VisualHallucinationAudit:
        """
        Prompts the SLM to check for obvious mismatches.
        Returns an Audit object containing the boolean result and an optional discard reason.
        """
        pass


class IVisualHallucinationRepository(ABC):
    """
    Contract for persisting the hallucination evaluation audit trail to PostgreSQL.
    """

    @abstractmethod
    def save(self, transcript: VisualHallucinationEvaluatedTranscript) -> VisualHallucinationEvaluatedTranscript:
        pass

    @abstractmethod
    def get_by_source_id(self, source_id: uuid.UUID) -> Optional[VisualHallucinationEvaluatedTranscript]:
        pass