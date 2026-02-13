import pytest
from sqlalchemy import text
from app.core.database.service.manager import db_manager
from app.core.database import exceptions as exc

@pytest.mark.integration
def test_transaction_rollback_on_failure():
    """
    Verify that a failed transaction in the real DB does not persist data.
    """
    # 1. We'll use a temporary table for this test to be safe
    with db_manager.session() as session:
        session.execute(text("CREATE TEMP TABLE test_rollback (id int)"))

    # 2. Try to insert data then fail
    try:
        with db_manager.session() as session:
            session.execute(text("INSERT INTO test_rollback VALUES (1)"))
            raise Exception("Force Failure")
    except Exception:
        pass # We expect this

    # 3. Verify data was NOT saved
    with db_manager.session() as session:
        result = session.execute(text("SELECT count(*) FROM test_rollback")).scalar()
        assert result == 0

@pytest.mark.integration
def test_real_schema_error_mapping():
    """
    Verify that querying a non-existent table in Postgres
    actually raises our custom DBSchemaError.
    """
    with pytest.raises(exc.DBSchemaError):
        with db_manager.session() as session:
            # This table definitely doesn't exist
            session.execute(text("SELECT * FROM non_existent_ghost_table"))

@pytest.mark.integration
def test_real_syntax_error_mapping():
    """
    Verify that bad SQL raises a Schema/Programming error.
    """
    with pytest.raises(exc.DBSchemaError):
        with db_manager.session() as session:
            # 'SELECKT' is a syntax error
            session.execute(text("SELECKT * FROM now()"))