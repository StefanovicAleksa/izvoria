from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from app.core.config import settings
from app.core.database import exceptions as exc

# The Base Class for all future models
class Base(DeclarativeBase):
    pass

class SQLAlchemyEngine:
    def __init__(self):
        self._engine = None
        self._session_maker = None
        self._initialize()

    def _initialize(self):
        try:
            self._engine = create_engine(
                settings.DATABASE_URL,
                # POOLING CONFIGURATION
                pool_pre_ping=True,              # Auto-detect disconnects
                pool_size=settings.DB_POOL_SIZE, # Keep 20 connections open
                max_overflow=settings.DB_MAX_OVERFLOW # Allow 10 more if busy
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

    def dispose_engine(self):
        if self._engine:
            self._engine.dispose()