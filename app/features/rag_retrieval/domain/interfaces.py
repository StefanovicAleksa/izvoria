# Canvas: rag_retrieval_interfaces_v1
# Path: app/features/rag_retrieval/domain/interfaces.py

from abc import ABC, abstractmethod
from typing import List, Optional
import uuid

from app.core.model_manager.domain.entities import LoadedModel
from app.core.model_manager.domain.interfaces import IModelLoader
from app.features.rag_retrieval.domain.entities import IndexedSource, RetrievedChunk

class IPromptEmbeddingAdapter(IModelLoader, ABC):
    """
    Contract for vectorizing a single user query.
    Inherits from IModelLoader so the Orchestrator can manage its VRAM lifecycle.
    """
    @abstractmethod
    def embed_prompt(self, model: LoadedModel, text: str) -> List[float]:
        """Converts the raw string prompt into a dense mathematical vector."""
        pass

    @abstractmethod
    def embed_prompts(self, model: LoadedModel, texts: List[str]) -> List[List[float]]:
        """
        Vectorizes several prompts in one forward pass, in the given order.

        This is what lets a query fan-out cost one model acquisition instead of one per query: the load and
        unload of the embedder dominate the embedding itself for short prompts.
        """
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
        source_id_filter: Optional[uuid.UUID] = None,
        dense_keep: int = 0
    ) -> List[RetrievedChunk]:
        """
        `dense_keep`: the dense leg's best this many are returned too, beyond `limit` if they fused lower -- for a
        pool the reranker will sort; 0 returns exactly the fused top `limit`.

        Executes both Semantic (Dense) and Keyword (Sparse) searches,
        merges them using Reciprocal Rank Fusion (RRF), and returns the top results.
        """
        pass

    @abstractmethod
    def list_indexed_sources(self, min_chunks: int = 1) -> List[IndexedSource]:
        """
        Every source with rows in `vector_chunks`, best-populated first, with its human-readable name.

        Storage's `list_sources()` answers "what was ingested"; this answers "what can actually be
        retrieved", which is the only honest basis for a claim about corpus coverage.
        """
        pass

    @abstractmethod
    def pick_sweep_sources(
        self,
        query_vector: List[float],
        ann_depth: int,
        max_sources: int
    ) -> List[uuid.UUID]:
        """
        The sources whose best chunk is nearest the query, via one index-served ANN probe.

        Deliberately not a `GROUP BY source_id` over the whole table with a per-group MIN(distance): no
        index can serve that, so it degrades into a full scan with a distance computation per row.
        """
        pass

    @abstractmethod
    def fetch_chunks(self, chunk_ids: List[uuid.UUID]) -> List[RetrievedChunk]:
        """
        The given chunks, in the given order (duplicates once). Ids that no longer exist are skipped.
        """
        pass

    @abstractmethod
    def fetch_source_chunks(self, source_id: uuid.UUID) -> List[RetrievedChunk]:
        """Every indexed chunk of one source, in reading order."""
        pass

    @abstractmethod
    def fetch_neighbors(self, chunk_ids: List[uuid.UUID]) -> List[RetrievedChunk]:
        """
        For each given chunk, the chunks immediately before and after it (by start time) in the same source.
        The given chunks themselves are not returned unless they neighbour another given chunk.
        """
        pass
