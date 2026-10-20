# Canvas: rag_storage_interfaces_v3
# Path: app/features/rag_storage/domain/interfaces.py

from abc import ABC, abstractmethod
from typing import List
from uuid import UUID

from app.core.model_manager.domain.entities import LoadedModel
from app.core.model_manager.domain.interfaces import IModelLoader
from app.features.context_unification.domain.entities import UnifiedChunk
from app.features.rag_storage.domain.entities import StoredChunkText


class IEmbeddingAdapter(IModelLoader, ABC):
    @abstractmethod
    def embed_batch(self, model: LoadedModel, texts: List[str]) -> List[List[float]]:
        pass


class IRagRepository(ABC):
    @abstractmethod
    def replace_rag_chunks(self, source_id: UUID, chunks: List[UnifiedChunk], embeddings: List[List[float]]) -> int:
        """
        Replaces every chunk of a source in a single transaction: the old dense + sparse rows
        are deleted and the new ones inserted atomically, so a failure never leaves a source
        half-indexed or empty.
        """
        pass

    @abstractmethod
    def delete_by_source_id(self, source_id: UUID) -> None:
        """Atomically deletes both vector and sparse data for a source."""
        pass

    @abstractmethod
    def read_chunk_texts(self, source_id: UUID) -> List[StoredChunkText]:
        """Every indexed chunk of a source with its stored context and raw text, in reading order."""
        pass

    @abstractmethod
    def update_chunk_payloads(self, source_id: UUID, chunks: List[StoredChunkText], payloads: List[str],
                              embeddings: List[List[float]]) -> int:
        """
        Rewrites the context, payload and embedding of existing chunks in one transaction, keeping their ids, so
        clips, conversation evidence and report citations that point at them stay valid. Returns the rows updated.
        """
        pass
