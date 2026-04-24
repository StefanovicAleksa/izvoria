# Canvas: rag_storage_interfaces_v2
# Path: app/features/rag_storage/domain/interfaces.py

from abc import ABC, abstractmethod
from typing import List
from uuid import UUID

from app.core.model_manager.domain.entities import LoadedModel
from app.core.model_manager.domain.interfaces import IModelLoader
from app.features.context_unification.domain.entities import UnifiedChunk


class IEmbeddingAdapter(IModelLoader, ABC):
    @abstractmethod
    def embed_batch(self, model: LoadedModel, texts: List[str]) -> List[List[float]]:
        pass


class IRagRepository(ABC):
    @abstractmethod
    def save_rag_chunks(self, source_id: UUID, chunks: List[UnifiedChunk], embeddings: List[List[float]]) -> int:
        """Saves both the dense vector embedding and the sparse FTS index atomically."""
        pass

    @abstractmethod
    def delete_by_source_id(self, source_id: UUID) -> None:
        """Atomically deletes both vector and sparse data for a source."""
        pass