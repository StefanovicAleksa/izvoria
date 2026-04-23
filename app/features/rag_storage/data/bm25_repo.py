# Canvas: bm25_repo_v2
# Path: app/features/rag_storage/data/bm25_repo.py

import os
import pickle
import logging
from typing import List, Dict, Any
from uuid import UUID
from rank_bm25 import BM25Okapi

from app.features.rag_storage.domain.interfaces import ISparseRepository
from app.features.context_unification.domain.entities import UnifiedChunk
from app.features.rag_storage.utils.error_mapper import map_storage_error

logger = logging.getLogger(__name__)


class LocalBM25Repository(ISparseRepository):
    """
    Concrete implementation of ISparseRepository.
    Uses rank_bm25 and stores the serialized index on the local disk per source.
    """

    def __init__(self):
        # We store BM25 indexes in a subfolder of our primary persistent data path
        # FIXED: Decoupled from the removed CHROMA_DB_PATH setting.
        self.index_dir = os.path.join(os.getcwd(), "data", "bm25_indexes")
        os.makedirs(self.index_dir, exist_ok=True)

    def _get_file_path(self, source_id: UUID) -> str:
        """Returns the strict local path for a given source's BM25 index."""
        return os.path.join(self.index_dir, f"{source_id}.pkl")

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenizer for BM25.
        Converts to lowercase and splits by whitespace/basic punctuation.
        """
        import re
        text = text.lower()
        return re.findall(r'\b\w+\b', text)

    def save_sparse_index(self, source_id: UUID, chunks: List[UnifiedChunk]) -> int:
        if not chunks:
            return 0

        try:
            # 1. Tokenize all chunk payloads
            tokenized_corpus = [self._tokenize(chunk.text_payload) for chunk in chunks]

            # 2. Fit the BM25 algorithm
            bm25 = BM25Okapi(tokenized_corpus)

            # 3. Create the payload containing the index and the chunk mappings
            payload: Dict[str, Any] = {
                "source_id": str(source_id),
                "bm25_index": bm25,
                "chunk_ids": [chunk.chunk_id for chunk in chunks]
            }

            # 4. Serialize to disk
            file_path = self._get_file_path(source_id)
            with open(file_path, "wb") as f:
                pickle.dump(payload, f)

            logger.info(f"Successfully saved BM25 sparse index for source {source_id}.")
            return len(chunks)

        except Exception as e:
            raise map_storage_error(e, context=f"Saving BM25 sparse index for {source_id}")

    def delete_by_source_id(self, source_id: UUID) -> None:
        """Idempotency check: Removes the serialized index if it exists."""
        file_path = self._get_file_path(source_id)
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleared existing BM25 sparse index for source {source_id}.")
        except Exception as e:
            raise map_storage_error(e, context=f"Clearing BM25 index for {source_id}")