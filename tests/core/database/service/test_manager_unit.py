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

class TestDatabaseSelection:
    """Without arguments the appliance's database; the hosted API passes a firm's URL and a smaller pool."""

    @pytest.fixture
    def create_engine(self, mocker):
        mocker.patch("app.core.database.data.engine.event")  # the password hook needs a real engine
        return mocker.patch("app.core.database.data.engine.create_engine")

    def test_defaults_to_the_appliance_database_and_pool(self, create_engine):
        from app.core.config import settings

        DatabaseManager()

        create_engine.assert_called_once_with(settings.DATABASE_URL, pool_pre_ping=True,
                                              pool_size=settings.DB_POOL_SIZE, max_overflow=settings.DB_MAX_OVERFLOW)

    def test_a_firm_database_with_its_own_pool(self, create_engine):
        DatabaseManager(url="postgresql://u:p@db:5432/onyx_firm_legal_demo", pool_size=3, max_overflow=1)

        create_engine.assert_called_once_with("postgresql://u:p@db:5432/onyx_firm_legal_demo", pool_pre_ping=True,
                                              pool_size=3, max_overflow=1)

    def test_a_zero_overflow_is_kept_not_replaced_by_the_default(self, create_engine):
        DatabaseManager(url="postgresql://x", pool_size=2, max_overflow=0)

        assert create_engine.call_args.kwargs["max_overflow"] == 0
