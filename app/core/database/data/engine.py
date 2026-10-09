from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from app.core.config import settings
from app.core.database import exceptions as exc

# The Base Class for all future models
class Base(DeclarativeBase):
    pass

class SQLAlchemyEngine:
    """
    One connection pool. Without arguments it is the appliance's own database (settings.DATABASE_URL); the hosted
    API passes a firm's database URL and a smaller pool, since it keeps one engine per firm.
    """

    def __init__(self, url: Optional[str] = None, pool_size: Optional[int] = None,
                 max_overflow: Optional[int] = None):
        self._url = url
        self._pool_size = pool_size
        self._max_overflow = max_overflow
        self._engine = None
        self._session_maker = None
        self._initialize()

    def _initialize(self):
        try:
            self._engine = create_engine(
                self._url or settings.DATABASE_URL,
                # POOLING CONFIGURATION
                pool_pre_ping=True,              # Auto-detect disconnects
                pool_size=self._pool_size if self._pool_size is not None else settings.DB_POOL_SIZE,
                max_overflow=self._max_overflow if self._max_overflow is not None else settings.DB_MAX_OVERFLOW
            )
            self._session_maker = sessionmaker(
                bind=self._engine,
                autocommit=False,
                autoflush=False
            )
        except Exception as e:
            raise exc.DBConnectionRefusedError(f"Engine Init Failed: {e}", original_exception=e)

    def get_session_maker(self):
        return self._session_maker

    def get_engine(self):
        return self._engine

    def dispose_engine(self):
        if self._engine:
            self._engine.dispose()