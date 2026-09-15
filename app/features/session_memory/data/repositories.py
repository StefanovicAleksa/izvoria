# Path: app/features/session_memory/data/repositories.py

import uuid
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import select

from app.core.database.domain.interfaces import IDatabaseManager
from app.core.database import exceptions as db_exc
from app.features.session_memory.domain.interfaces import ISessionMemoryRepository
from app.features.session_memory.domain.entities import SessionTurn
from app.features.session_memory.data.models import SessionTurnModel
from app.features.session_memory.utils.error_mapper import map_session_memory_error


class SessionMemoryRepository(ISessionMemoryRepository):
    def __init__(self, db_manager: IDatabaseManager):
        self._db = db_manager

    def record_turn(self, session_id: uuid.UUID, question: str, answer_text: str) -> SessionTurn:
        try:
            with self._db.session() as session:
                turn_model = SessionTurnModel(
                    id=uuid.uuid4(),
                    session_id=session_id,
                    question=question,
                    answer_text=answer_text,
                    created_at=datetime.now(timezone.utc)
                )
                session.add(turn_model)
                return SessionTurn.model_validate(turn_model)

        except db_exc.DatabaseError as e:
            raise map_session_memory_error(e)

    def get_last_turn(self, session_id: uuid.UUID) -> Optional[SessionTurn]:
        try:
            with self._db.session() as session:
                stmt = (
                    select(SessionTurnModel)
                    .where(SessionTurnModel.session_id == session_id)
                    .order_by(SessionTurnModel.created_at.desc())
                    .limit(1)
                )
                model = session.execute(stmt).scalars().first()
                return SessionTurn.model_validate(model) if model else None

        except db_exc.DatabaseError as e:
            raise map_session_memory_error(e)
