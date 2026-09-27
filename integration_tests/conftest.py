# Path: integration_tests/conftest.py

import os
import pytest
import shutil
import tempfile
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

# Glossaries are seeded into a throwaway directory. Copying test resources into data/glossaries overwrote the
# application's real glossaries (e.g. the options-trading finance.json) on every run.
_TEST_GLOSSARY_DIR = tempfile.mkdtemp(prefix="onyx_test_glossaries_")
os.environ["GLOSSARY_PATH"] = _TEST_GLOSSARY_DIR

from app.core.config import settings
from app.core.database.service.manager import db_manager as core_db_manager
from app.core.database.service.migrations import upgrade_to_head
from app.core.database.utils.model_registry import import_all_models

# Register every feature's tables. Otherwise a table exists only if some collected test file happens to import its
# model (e.g. hybrid search joins `video_clips`, so live pipeline tests failed when run on their own).
import_all_models()


@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    """
    1. Waits for the Database to become responsive.
    2. Forcefully drops the public schema to bypass Foreign Key deadlocks.
    3. Re-enables pgvector extension.
    4. Builds the schema through the migrations, exactly as OnyxEngine.start() does on an appliance.
    5. Seeds dummy glossaries into the throwaway GLOSSARY_PATH directory (never data/glossaries).
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

    # C. BUILD THE SCHEMA THROUGH THE MIGRATIONS
    upgrade_to_head(real_engine)

    # D. SEED MOCK GLOSSARIES FROM TEST RESOURCES
    # Copies your dummy json files so the tests don't crash looking for them
    source_glossaries = settings.BASE_DIR / "integration_tests" / "resources"
    test_glossaries = settings.GLOSSARY_DIR
    test_glossaries.mkdir(parents=True, exist_ok=True)

    if source_glossaries.exists():
        for item in source_glossaries.glob("*.json"):
            shutil.copy2(item, test_glossaries / item.name)

    yield

    shutil.rmtree(_TEST_GLOSSARY_DIR, ignore_errors=True)


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