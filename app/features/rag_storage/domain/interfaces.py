from abc import ABC, abstractmethod
from typing import List
from uuid import UUID

from app.core.model_manager.domain.entities import LoadedModel
from app.core.model_manager.domain.interfaces import IModelLoader
from app.features.context_unification.domain.entities import UnifiedChunk


# Inherit from IModelLoader so the Orchestrator can securely manage it
class IEmbeddingAdapter(IModelLoader, ABC):
    @abstractmethod
    def embed_batch(self, model: LoadedModel, texts: List[str]) -> List[List[float]]:
        pass


class IDenseRepository(ABC):
    @abstractmethod
    def save_dense_vectors(self, source_id: UUID, chunks: List[UnifiedChunk], embeddings: List[List[float]]) -> int:
        pass

    @abstractmethod
    def delete_by_source_id(self, source_id: UUID) -> None:
        pass


class ISparseRepository(ABC):
    @abstractmethod
    def save_sparse_index(self, source_id: UUID, chunks: List[UnifiedChunk]) -> int:
        pass

    @abstractmethod
    def delete_by_source_id(self, source_id: UUID) -> None:
        pass