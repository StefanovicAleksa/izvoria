import pytest
import time
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from app.core.database.service.manager import db_manager

@pytest.fixture(scope="session", autouse=True)
def wait_for_db():
    """
    Waits for the Database to become responsive before running integration tests.
    This prevents 'Connection Refused' errors if Docker is still starting up.
    """
    retries = 5
    for i in range(retries):
        try:
            # Try to grab a session and run a simple query
            with db_manager.session() as session:
                session.execute(text("SELECT 1"))
            return # Success!
        except (OperationalError, Exception):
            if i == retries - 1:
                pytest.fail("Could not connect to the Integration Test Database. Is Docker running?")
            time.sleep(1) # Wait 1s before retrying