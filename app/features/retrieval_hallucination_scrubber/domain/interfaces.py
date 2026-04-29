# Canvas: retrieval_scrubber_interfaces_v2
# Path: app/features/retrieval_hallucination_scrubber/domain/interfaces.py

import uuid
from abc import ABC, abstractmethod

from app.core.model_manager.domain.interfaces import IModelLoader
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.features.retrieval_hallucination_scrubber.domain.entities import ScrubAudit

class IRetrievalScrubberAdapter(IModelLoader, ABC):
    """
    Contract for the SLM (e.g., Qwen 2.5 3B) that acts strictly as the RAG context gatekeeper.
    """

    @abstractmethod
    def load(self, manifest: ModelManifest) -> LoadedModel:
        """Loads the SLM into VRAM."""
        pass

    @abstractmethod
    def unload(self, model: LoadedModel) -> bool:
        """Safely removes the SLM from VRAM and triggers garbage collection."""
        pass

    @abstractmethod
    def evaluate_relevance(
        self,
        model: LoadedModel,
        chunk_id: uuid.UUID,
        user_prompt: str,
        text_chunk: str
    ) -> ScrubAudit:
        """
        Prompts the SLM to verify if the text_chunk is relevant to the user_prompt.
        Returns a ScrubAudit with the boolean result and reason.
        """
        pass