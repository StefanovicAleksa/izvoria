from contextlib import contextmanager
from typing import Generator
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from app.core.database.domain.interfaces import IDatabaseManager
from app.core.database.data.engine import SQLAlchemyEngine
from app.core.database.utils.error_mapper import map_database_error
from app.core.database import exceptions as exc

class DatabaseManager(IDatabaseManager):
    def __init__(self):
        self._engine_wrapper = SQLAlchemyEngine()
        self._session_factory = self._engine_wrapper.get_session_maker()

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """
        Transactional Scope with Automatic Error Translation.
        """
        session: Session = self._session_factory()
        try:
            yield session
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            # The Magic: Translate the cryptic SQL error to a Domain Error
            raise map_database_error(e) from e
        except Exception as e:
            session.rollback()
            # Catch non-SQL errors (like Pydantic validation inside the block)
            raise exc.SystemError(f"Unexpected System Error: {str(e)}", original_exception=e) from e
        finally:
            session.close()

    def dispose(self) -> None:
        self._engine_wrapper.dispose_engine()

# Singleton Instance
db_manager = DatabaseManager()