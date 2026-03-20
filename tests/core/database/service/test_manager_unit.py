import pytest
from unittest.mock import MagicMock
from sqlalchemy.exc import SQLAlchemyError
from app.core.database.service.manager import DatabaseManager
from app.core.database import exceptions as exc


@pytest.fixture
def mock_db_manager(mocker):
    """
    Creates a DatabaseManager with a mocked Engine and Session.
    """
    # 1. Mock the SQLAlchemyEngine class
    mock_engine_cls = mocker.patch("app.core.database.service.manager.SQLAlchemyEngine")

    # 2. Setup the mock session
    mock_session = MagicMock()

    # 3. Configure the engine to return our mock session factory
    mock_engine_instance = mock_engine_cls.return_value
    # The factory returns the session instance
    mock_engine_instance.get_session_maker.return_value = MagicMock(return_value=mock_session)

    # 4. Initialize the manager
    manager = DatabaseManager()

    return manager, mock_session


def test_session_commit_on_success(mock_db_manager):
    """Verify that if code runs without error, session.commit() is called."""
    manager, mock_session = mock_db_manager

    # Use '_' because we don't need the session object in this test
    with manager.session() as _:
        pass  # Do nothing, just exit successfully

    # Assertions
    mock_session.commit.assert_called_once()
    mock_session.close.assert_called_once()
    mock_session.rollback.assert_not_called()


def test_session_rollback_on_error(mock_db_manager):
    """Verify that if an error occurs, session.rollback() is called."""
    manager, mock_session = mock_db_manager

    # We expect SystemError because we raise a generic Exception below
    with pytest.raises(exc.SystemError):
        with manager.session() as _:
            raise Exception("Something went wrong!")

    # Assertions
    mock_session.rollback.assert_called_once()
    mock_session.close.assert_called_once()
    mock_session.commit.assert_not_called()


def test_session_translates_sqlalchemy_error(mock_db_manager):
    """Verify that SQLAlchemy errors are caught and translated."""
    manager, mock_session = mock_db_manager

    # Simulate a DB error when commit() is called
    mock_session.commit.side_effect = SQLAlchemyError("DB Exploded")

    # We expect our mapped DatabaseError
    with pytest.raises(exc.DatabaseError):
        with manager.session() as _:
            pass  # The error happens at the end (during commit)

    mock_session.rollback.assert_called_once()