# Path: integration_tests/conftest.py

import os
import pytest
import shutil
import time
from pathlib import Path
from contextlib import contextmanager
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError

# ==========================================
# CRITICAL: ENVIRONMENT DECOUPLING
# ==========================================
# Force the application into 'test' mode BEFORE any app modules are imported.
# This guarantees config.py evaluates ENV="test" on startup.
os.environ["ENV"] = "test"

from app.core.config import settings
from app.core.database.service.manager import db_manager as core_db_manager
from app.core.database.data.engine import Base

# Import models so SQLAlchemy "sees" them for create_all()
from app.features.storage.data.models import SourceModel, FileModel  # noqa
from app.core.jobs.data.models import JobModel  # noqa
from app.features.rag_storage.data.models import VectorChunkModel # noqa

# Register every feature's tables too. Otherwise a table exists only if some collected test file happens to
# import its model (e.g. hybrid search joins `video_clips`, so live pipeline tests failed when run on their own).
import importlib  # noqa: E402

for _models_file in sorted(settings.BASE_DIR.glob("app/**/data/models.py")):
    importlib.import_module(".".join(_models_file.relative_to(settings.BASE_DIR).with_suffix("").parts))


@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    """
    1. Waits for the Database to become responsive.
    2. Forcefully drops the public schema to bypass Foreign Key deadlocks.
    3. Re-enables pgvector extension.
    4. Creates all tables (schema) required for the integration tests.
    5. Seeds dummy glossaries for domain adaptation tests.
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

    # B. WIPE THE STALE SCHEMA COMPLETELY & ENABLE VECTOR
    with real_engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE;"))
        conn.execute(text("CREATE SCHEMA public;"))

        # CRITICAL ADDITION: Re-enable pgvector after wiping the schema
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))

    # C. CREATE ALL TABLES WITH THE NEW SCHEMA
    Base.metadata.create_all(bind=real_engine)

    # D. SEED MOCK GLOSSARIES FROM TEST RESOURCES
    # Copies your dummy json files so the tests don't crash looking for them
    source_glossaries = settings.BASE_DIR / "integration_tests" / "resources"
    test_glossaries = settings.GLOSSARY_DIR
    test_glossaries.mkdir(parents=True, exist_ok=True)

    if source_glossaries.exists():
        for item in source_glossaries.glob("*.json"):
            shutil.copy2(item, test_glossaries / item.name)

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
        # FIX: Use a nested transaction (SAVEPOINT) to mimic real commits!
        # This forces SQLAlchemy to flush queries sequentially to Postgres
        # exactly as the app intends, fixing our Foreign Key constraint crashes.
        nested = session.begin_nested()
        try:
            yield session
            nested.commit()  # Flushes the SQL to the DB without breaking test isolation
        except Exception:
            nested.rollback()
            raise

    core_db_manager.session = test_session_override

    yield core_db_manager

    core_db_manager.session = original_session_method

    transaction.rollback()
    session.close()
    connection.close()