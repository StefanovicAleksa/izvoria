# Canvas: rag_retrieval_interfaces_v1
# Path: app/features/rag_retrieval/domain/interfaces.py

from abc import ABC, abstractmethod
from typing import List, Optional
import uuid

from app.core.model_manager.domain.entities import LoadedModel
from app.core.model_manager.domain.interfaces import IModelLoader
from app.features.rag_retrieval.domain.entities import RetrievedChunk

class IPromptEmbeddingAdapter(IModelLoader, ABC):
    """
    Contract for vectorizing a single user query.
    Inherits from IModelLoader so the Orchestrator can manage its VRAM lifecycle.
    """
    @abstractmethod
    def embed_prompt(self, model: LoadedModel, text: str) -> List[float]:
        """Converts the raw string prompt into a dense mathematical vector."""
        pass

class IRetrievalRepository(ABC):
    """
    Contract for the database search execution.
    Ensures the service layer is decoupled from pgvector/Postgres specifics.
    """
    @abstractmethod
    def execute_hybrid_search(
        self,
        query_vector: List[float],
        query_text: str,
        limit: int,
        source_id_filter: Optional[uuid.UUID] = None
    ) -> List[RetrievedChunk]:
        """
        Executes both Semantic (Dense) and Keyword (Sparse) searches,
        merges them using Reciprocal Rank Fusion (RRF), and returns the top results.
        """
        pass

    @abstractmethod
    def fetch_chunks(self, chunk_ids: List[uuid.UUID]) -> List[RetrievedChunk]:
        """
        The given chunks, in the given order (duplicates once). Ids that no longer exist are skipped.
        """
        pass

    @abstractmethod
    def fetch_neighbors(self, chunk_ids: List[uuid.UUID]) -> List[RetrievedChunk]:
        """
        For each given chunk, the chunks immediately before and after it (by start time) in the same source.
        The given chunks themselves are not returned unless they neighbour another given chunk.
        """
        pass
