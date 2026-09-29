# Path: app/features/search_history/service/search_history_api.py

import uuid
from typing import List, Optional

from app.features.search_history.domain.entities import SearchRecord, SearchRecordSummary
from app.features.search_history.domain.enums import SearchHistoryMode
from app.features.search_history.domain.interfaces import ISearchHistoryRepository
from app.features.search_history.utils.error_mapper import map_search_history_error

DEFAULT_LIST_LIMIT = 200


class SearchHistoryAPI:
    """
    What was searched for and what it returned, so a past search can be reopened exactly as it was shown. Reading
    a record loads no model: the hits are chunk ids, hydrated by the caller.
    """

    def __init__(self, repository: ISearchHistoryRepository):
        self._repo = repository

    def record(
        self,
        query: str,
        mode: SearchHistoryMode,
        result_limit: int,
        chunk_ids: List[uuid.UUID],
        scores: List[float],
        execution_time_ms: float,
    ) -> SearchRecord:
        try:
            return self._repo.record(query, mode, result_limit, list(chunk_ids), list(scores), execution_time_ms)
        except Exception as e:
            raise map_search_history_error(e)

    def list(self, limit: int = DEFAULT_LIST_LIMIT) -> List[SearchRecordSummary]:
        try:
            return self._repo.list(limit)
        except Exception as e:
            raise map_search_history_error(e)

    def get(self, record_id: uuid.UUID) -> Optional[SearchRecord]:
        try:
            return self._repo.get(record_id)
        except Exception as e:
            raise map_search_history_error(e)

    def delete(self, record_id: uuid.UUID) -> bool:
        try:
            return self._repo.delete(record_id)
        except Exception as e:
            raise map_search_history_error(e)
