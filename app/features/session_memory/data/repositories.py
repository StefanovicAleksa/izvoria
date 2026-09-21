# Path: app/features/session_memory/data/repositories.py

import uuid
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import select

from app.core.database.domain.interfaces import IDatabaseManager
from app.core.database import exceptions as db_exc
from app.features.session_memory.domain.interfaces import ISessionMemoryRepository
from app.features.session_memory.domain.entities import SessionTurn
from app.features.session_memory.domain.enums import SessionTurnMode
from app.features.session_memory.data.models import SessionTurnModel
from app.features.session_memory.utils.error_mapper import map_session_memory_error


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
                return [SessionTurn.model_validate(m) for m in reversed(models)]

        except db_exc.DatabaseError as e:
            raise map_session_memory_error(e)
