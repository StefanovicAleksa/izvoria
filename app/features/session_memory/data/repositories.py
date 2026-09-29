# Path: app/features/session_memory/data/repositories.py

import logging
import uuid
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import Text, delete, func, select
from sqlalchemy.dialects.postgresql import ARRAY, aggregate_order_by

from app.core.database.domain.interfaces import IDatabaseManager
from app.core.database import exceptions as db_exc
from app.features.session_memory.domain.interfaces import ISessionMemoryRepository
from app.features.session_memory.domain.entities import SessionSummary, SessionTurn
from app.features.session_memory.domain.enums import SessionTurnMode
from app.features.session_memory.data.models import SessionTurnModel
from app.features.session_memory.utils.error_mapper import map_session_memory_error

logger = logging.getLogger(__name__)


def _to_turn(model: SessionTurnModel) -> SessionTurn:
    """
    A stored `mode` this build does not know about degrades to CHAT instead of raising. Modes have been
    renamed once already ('deep_research' -> 'reasoning'), and the pre-Alembic catch-up backfills that on
    boot — but a database reached without going through OnyxEngine.start() would otherwise fail validation
    and take the entire conversation down over a label. Losing which entrypoint answered a turn is survivable;
    losing the turn is not.
    """
    try:
        SessionTurnMode(model.mode)
    except ValueError:
        logger.warning(
            f"Session turn {model.id} has unrecognised mode {model.mode!r}; reading it as "
            f"{SessionTurnMode.CHAT.value!r}. Has this database been migrated (OnyxEngine.start)?"
        )
        return SessionTurn(
            id=model.id,
            session_id=model.session_id,
            question=model.question,
            answer_text=model.answer_text,
            mode=SessionTurnMode.CHAT,
            standalone_query=model.standalone_query,
            evidence_chunk_ids=list(model.evidence_chunk_ids or []),
            first_doc_number=model.first_doc_number,
            created_at=model.created_at,
        )

    return SessionTurn.model_validate(model)


class SessionMemoryRepository(ISessionMemoryRepository):
    def __init__(self, db_manager: IDatabaseManager):
        self._db = db_manager

    def record_turn(
        self,
        session_id: uuid.UUID,
        question: str,
        answer_text: str,
        mode: SessionTurnMode = SessionTurnMode.CHAT,
        standalone_query: Optional[str] = None,
        evidence_chunk_ids: Optional[List[uuid.UUID]] = None,
        first_doc_number: int = 1
    ) -> SessionTurn:
        try:
            with self._db.session() as session:
                turn_model = SessionTurnModel(
                    id=uuid.uuid4(),
                    session_id=session_id,
                    question=question,
                    answer_text=answer_text,
                    mode=mode.value,
                    standalone_query=standalone_query,
                    evidence_chunk_ids=list(evidence_chunk_ids or []),
                    first_doc_number=first_doc_number,
                    created_at=datetime.now(timezone.utc)
                )
                session.add(turn_model)
                return SessionTurn.model_validate(turn_model)

        except db_exc.DatabaseError as e:
            raise map_session_memory_error(e)

    def get_last_turn(self, session_id: uuid.UUID) -> Optional[SessionTurn]:
        turns = self.get_history(session_id, limit=1)
        return turns[0] if turns else None

    def get_history(self, session_id: uuid.UUID, limit: int) -> List[SessionTurn]:
        try:
            with self._db.session() as session:
                stmt = (
                    select(SessionTurnModel)
                    .where(SessionTurnModel.session_id == session_id)
                    .order_by(SessionTurnModel.created_at.desc())
                    .limit(limit)
                )
                models = session.execute(stmt).scalars().all()
                return [_to_turn(m) for m in reversed(models)]

        except db_exc.DatabaseError as e:
            raise map_session_memory_error(e)

    def get_all_turns(self, session_id: uuid.UUID) -> List[SessionTurn]:
        try:
            with self._db.session() as session:
                stmt = (
                    select(SessionTurnModel)
                    .where(SessionTurnModel.session_id == session_id)
                    .order_by(SessionTurnModel.created_at.asc())
                )
                return [_to_turn(m) for m in session.execute(stmt).scalars().all()]

        except db_exc.DatabaseError as e:
            raise map_session_memory_error(e)

    def list_sessions(self, limit: int) -> List[SessionSummary]:
        try:
            with self._db.session() as session:
                first_question = func.array_agg(
                    aggregate_order_by(SessionTurnModel.question, SessionTurnModel.created_at.asc()),
                    type_=ARRAY(Text),
                )[1]
                last_at = func.max(SessionTurnModel.created_at)
                stmt = (
                    select(
                        SessionTurnModel.session_id,
                        first_question.label("title"),
                        func.min(SessionTurnModel.created_at).label("started_at"),
                        last_at.label("last_at"),
                        func.count().label("turn_count"),
                    )
                    .group_by(SessionTurnModel.session_id)
                    .order_by(last_at.desc())
                    .limit(limit)
                )
                return [
                    SessionSummary(
                        session_id=row.session_id,
                        title=row.title or "Untitled conversation",
                        started_at=row.started_at,
                        last_at=row.last_at,
                        turn_count=int(row.turn_count),
                    )
                    for row in session.execute(stmt).all()
                ]

        except db_exc.DatabaseError as e:
            raise map_session_memory_error(e)

    def delete_session(self, session_id: uuid.UUID) -> int:
        try:
            with self._db.session() as session:
                result = session.execute(delete(SessionTurnModel).where(SessionTurnModel.session_id == session_id))
                return int(result.rowcount or 0)

        except db_exc.DatabaseError as e:
            raise map_session_memory_error(e)
