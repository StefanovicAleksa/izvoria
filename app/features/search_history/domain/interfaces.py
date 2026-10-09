# Path: app/features/search_history/domain/interfaces.py

import uuid
from abc import ABC, abstractmethod
from typing import List, Optional

from app.features.search_history.domain.entities import SearchRecord, SearchRecordSummary
from app.features.search_history.domain.enums import SearchHistoryMode


class ISearchHistoryRepository(ABC):
    """Contract for persisting past searches to PostgreSQL."""

    @abstractmethod
    def record(
        self,
        query: str,
        mode: SearchHistoryMode,
        result_limit: int,
        chunk_ids: List[uuid.UUID],
        scores: List[float],
        execution_time_ms: float,
        owner_id: Optional[str] = None,
    ) -> SearchRecord:
        """Persists one search and its ranked results."""
        pass

    @abstractmethod
    def list(self, limit: int, owner_id: Optional[str] = None) -> List[SearchRecordSummary]:
        """The most recent `limit` searches, newest first (only `owner_id`'s, when given)."""
        pass

    @abstractmethod
    def get(self, record_id: uuid.UUID, owner_id: Optional[str] = None) -> Optional[SearchRecord]:
        """One search with its results, or None if it does not exist."""
        pass

    @abstractmethod
    def delete(self, record_id: uuid.UUID, owner_id: Optional[str] = None) -> bool:
        """Removes one search; says whether it existed."""
        pass
