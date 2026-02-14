import pytest
import time
from contextlib import contextmanager
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError

# FIX 1: Rename the import so it doesn't collide with our fixture name
from app.core.database.service.manager import db_manager as core_db_manager
# FIX 2: Import the Base class to access metadata for table creation
from app.core.database.data.engine import Base
# FIX 3: Crucial - Import models so SQLAlchemy "sees" them for create_all()
from app.features.storage.data.models import SourceModel, FileModel  # noqa


@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    """
    1. Waits for the Database to become responsive.
    2. Creates all tables (schema) required for the integration tests.
    """
    # Access the real engine through the wrapper
    # Manager -> SQLAlchemyEngine (_engine_wrapper) -> Real Engine (_engine)
    real_engine = core_db_manager._engine_wrapper._engine  # noqa

    # A. Wait for the DB to be ready
    retries = 5
    for i in range(retries):
        try:
            with real_engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            break  # Success!
        except (OperationalError, Exception):
            if i == retries - 1:
                pytest.fail("Could not connect to the Test Database. Is Docker running?")
            time.sleep(1)

    # B. CREATE ALL TABLES (The missing step that caused UndefinedTable)
    # This ensures "sources" and "files" exist in the DB
    Base.metadata.create_all(bind=real_engine)

    yield

    # Optional: Base.metadata.drop_all(bind=real_engine)
    # (Commented out to keep DB persistent between runs if desired)


@pytest.fixture(scope="function")
def db_manager():
    """
    Creates a transaction-aware database manager for integration tests.
    Rolls back changes after every test to maintain isolation.
    """
    real_engine = core_db_manager._engine_wrapper._engine  # noqa

    # 1. Start a connection and a transaction
    connection = real_engine.connect()
    transaction = connection.begin()

    # 2. Create session factory bound to this specific connection
    session_factory = sessionmaker(bind=connection)
    session = session_factory()

    # 3. Save original manager method
    original_session_method = core_db_manager.session

    # 4. Create the session override
    @contextmanager
    def test_session_override():
        yield session

    # 5. Monkeypatch the GLOBAL manager object
    core_db_manager.session = test_session_override

    # 6. Yield the patched manager to the test
    yield core_db_manager

    # 7. CLEANUP: Restore original method and rollback the transaction
    core_db_manager.session = original_session_method
    session.close()
    transaction.rollback()
    connection.close()