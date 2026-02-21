import pytest
import time
from contextlib import contextmanager
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError

from app.core.database.service.manager import db_manager as core_db_manager
from app.core.database.data.engine import Base

# Crucial - Import models so SQLAlchemy "sees" them for create_all()
from app.features.storage.data.models import SourceModel, FileModel  # noqa
from app.core.jobs.data.models import JobModel  # noqa (Include any other models you have!)


@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    """
    1. Waits for the Database to become responsive.
    2. Forcefully drops the public schema to bypass Foreign Key deadlocks.
    3. Creates all tables (schema) required for the integration tests.
    """
    real_engine = core_db_manager._engine_wrapper._engine  # noqa

    # A. Wait for the DB to be ready
    retries = 5
    for i in range(retries):
        try:
            with real_engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            break
        except (OperationalError, Exception):
            if i == retries - 1:
                pytest.fail("Could not connect to the Test Database. Is Docker running?")
            time.sleep(1)

    # B. WIPE THE STALE SCHEMA COMPLETELY (The Nuclear Option)
    # This completely bypasses the Foreign Key deadlock that drop_all() caused.
    with real_engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE;"))
        conn.execute(text("CREATE SCHEMA public;"))

    # C. CREATE ALL TABLES WITH THE NEW SCHEMA
    Base.metadata.create_all(bind=real_engine)

    yield


@pytest.fixture(scope="function")
def db_manager():
    """
    Creates a transaction-aware database manager for integration tests.
    """
    real_engine = core_db_manager._engine_wrapper._engine  # noqa

    connection = real_engine.connect()
    transaction = connection.begin()

    session_factory = sessionmaker(bind=connection)
    session = session_factory()

    original_session_method = core_db_manager.session

    @contextmanager
    def test_session_override():
        yield session

    core_db_manager.session = test_session_override

    yield core_db_manager

    core_db_manager.session = original_session_method

    transaction.rollback()
    session.close()
    connection.close()