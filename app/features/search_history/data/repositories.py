# Path: app/features/search_history/data/repositories.py

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import delete, func, select

from app.core.database.domain.interfaces import IDatabaseManager
from app.core.database import exceptions as db_exc
from app.features.search_history.data.models import SearchHistoryModel
from app.features.search_history.domain.entities import SearchRecord, SearchRecordSummary
from app.features.search_history.domain.enums import SearchHistoryMode
from app.features.search_history.domain.interfaces import ISearchHistoryRepository
from app.features.search_history.utils.error_mapper import map_search_history_error


def _to_record(model: SearchHistoryModel) -> SearchRecord:
    return SearchRecord(
        id=model.id,
        query=model.query,
        mode=SearchHistoryMode(model.mode),
        result_limit=model.result_limit,
        chunk_ids=list(model.chunk_ids or []),
        scores=[float(s) for s in (model.scores or [])],
        execution_time_ms=float(model.execution_time_ms or 0.0),
        created_at=model.created_at,
    )


def _owned(stmt, owner_id: Optional[str]):
    """Restricts a statement to one user's searches (the hosted API); None leaves it unrestricted (the appliance)."""
    return stmt if owner_id is None else stmt.where(SearchHistoryModel.owner_id == owner_id)


class SearchHistoryRepository(ISearchHistoryRepository):
    def __init__(self, db_manager: IDatabaseManager):
        self._db = db_manager

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
        try:
            with self._db.session() as session:
                model = SearchHistoryModel(
                    id=uuid.uuid4(),
                    query=query,
                    mode=mode.value,
                    result_limit=result_limit,
                    chunk_ids=list(chunk_ids),
                    scores=[float(s) for s in scores],
                    execution_time_ms=float(execution_time_ms),
                    created_at=datetime.now(timezone.utc),
                    owner_id=owner_id,
                )
                record = _to_record(model)  # validates before anything is written
                session.add(model)
                return record
        except db_exc.DatabaseError as e:
            raise map_search_history_error(e)

    def list(self, limit: int, owner_id: Optional[str] = None) -> List[SearchRecordSummary]:
        try:
            with self._db.session() as session:
                stmt = (
                    _owned(select(
                        SearchHistoryModel.id,
                        SearchHistoryModel.query,
                        SearchHistoryModel.mode,
                        func.cardinality(SearchHistoryModel.chunk_ids).label("hit_count"),
                        SearchHistoryModel.created_at,
                    ), owner_id)
                    .order_by(SearchHistoryModel.created_at.desc())
                    .limit(limit)
                )
                return [
                    SearchRecordSummary(
                        id=row.id, query=row.query, mode=SearchHistoryMode(row.mode),
                        hit_count=int(row.hit_count or 0), created_at=row.created_at,
                    )
                    for row in session.execute(stmt).all()
                ]
        except db_exc.DatabaseError as e:
            raise map_search_history_error(e)

    def get(self, record_id: uuid.UUID, owner_id: Optional[str] = None) -> Optional[SearchRecord]:
        try:
            with self._db.session() as session:
                model = session.get(SearchHistoryModel, record_id)
                if model is None or (owner_id is not None and model.owner_id != owner_id):
                    return None
                return _to_record(model)
        except db_exc.DatabaseError as e:
            raise map_search_history_error(e)

    def delete(self, record_id: uuid.UUID, owner_id: Optional[str] = None) -> bool:
        try:
            with self._db.session() as session:
                result = session.execute(
                    _owned(delete(SearchHistoryModel), owner_id).where(SearchHistoryModel.id == record_id))
                return bool(result.rowcount)
        except db_exc.DatabaseError as e:
            raise map_search_history_error(e)
