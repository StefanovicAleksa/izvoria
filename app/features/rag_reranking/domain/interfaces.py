# Path: app/features/rag_reranking/domain/interfaces.py

from abc import ABC, abstractmethod
from typing import List

from app.core.model_manager.domain.entities import LoadedModel
from app.core.model_manager.domain.interfaces import IModelLoader


class IRerankerAdapter(IModelLoader, ABC):
    """
    Contract for a cross-encoder that scores (query, chunk) pairs jointly.
    Inherits from IModelLoader so the Orchestrator manages its VRAM lifecycle.
    """

    @abstractmethod
    def score(self, model: LoadedModel, query: str, texts: List[str]) -> List[float]:
        """Returns one relevance score per text, in the same order. Higher means more relevant."""
        pass
