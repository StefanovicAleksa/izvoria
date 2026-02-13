from sqlalchemy.exc import IntegrityError, OperationalError
from app.core.database.utils.error_mapper import map_database_error
from app.core.database import exceptions as exc

# 1. Helper Class that mimics a real Psycopg2 Exception
class MockOrig(Exception):
    def __init__(self, code):
        self.pgcode = code
        # Mocking the 'diag' attribute which holds details
        self.diag = type('obj', (object,), {'message_detail': 'Mock Detail'})()

def create_pg_error(pgcode: str, message: str = "Mock Error"):
    """Creates a SQLAlchemy IntegrityError wrapping a Postgres error."""
    # We pass None for params and the original exception to satisfy the signature
    # But we overwrite .orig manually below
    e = IntegrityError(message, params={}, orig=MockOrig(pgcode))
    return e

def test_map_unique_violation():
    """Test that Postgres code 23505 becomes DBUniqueError"""
    raw_error = create_pg_error("23505")
    mapped_error = map_database_error(raw_error)
    assert isinstance(mapped_error, exc.DBUniqueError)

def test_map_foreign_key_violation():
    """Test that Postgres code 23503 becomes DBForeignKeyError"""
    raw_error = create_pg_error("23503")
    mapped_error = map_database_error(raw_error)
    assert isinstance(mapped_error, exc.DBForeignKeyError)

def test_map_connection_lost():
    """Test that Postgres code 08006 becomes DBUnreachableError"""
    # OperationalError is different from IntegrityError, so we mock it specifically
    e = OperationalError("Connection lost", params={}, orig=MockOrig("08006"))
    mapped_error = map_database_error(e)
    assert isinstance(mapped_error, exc.DBUnreachableError)

def test_map_missing_table():
    """Test that Postgres code 42P01 becomes DBSchemaError"""
    raw_error = create_pg_error("42P01")
    mapped_error = map_database_error(raw_error)
    assert isinstance(mapped_error, exc.DBSchemaError)

def test_fallback_unknown_error():
    """Test that unknown codes fall back to generic DatabaseError"""
    raw_error = create_pg_error("99999")
    mapped_error = map_database_error(raw_error)
    assert isinstance(mapped_error, exc.DatabaseError)