import pytest
from sqlalchemy import text
from app.core.database.service.manager import db_manager


@pytest.mark.integration
def test_database_connection():
    """
    Simple smoke test: Can we send SQL and get a result?
    """
    with db_manager.session() as session:
        result = session.execute(text("SELECT 'Onyx Rules'")).scalar()
        assert result == "Onyx Rules"


@pytest.mark.integration
def test_session_lifecycle():
    """
    Verify that the session context manager opens and closes correctly
    without throwing errors on a real backend.
    """
    with db_manager.session() as session:
        assert session.is_active
        # Do nothing, just exit

    # We can't strictly check 'session.is_active' here because the session
    # object is closed and destroyed, but if no error was raised, we are good.