from sqlalchemy.exc import SQLAlchemyError, OperationalError, IntegrityError, ProgrammingError, DataError as SADataError
import psycopg2.errors

from app.core.database import exceptions as exc


def map_database_error(sa_error: SQLAlchemyError) -> exc.DatabaseError:
    """
    Analyzes a SQLAlchemy Exception and converts it into a typed Izvoria Domain Exception.
    """
    # 1. Unwrap the exception to find the root cause (psycopg2 exception)
    original = getattr(sa_error, "orig", None)
    pg_code = getattr(original, "pgcode", None) if original else None
    error_msg = str(sa_error)

    # 2. Map by Postgres SQLSTATE Code
    # https://www.postgresql.org/docs/current/errcodes-appendix.html

    # Class 23: Integrity Constraint Violation
    if pg_code == "23505":  # unique_violation
        return exc.DBUniqueError("Duplicate entry detected.", original_exception=sa_error)
    if pg_code == "23503":  # foreign_key_violation
        return exc.DBForeignKeyError("Referenced record not found.", original_exception=sa_error)
    if pg_code == "23502":  # not_null_violation
        return exc.DBNotNullError("Missing required field.", original_exception=sa_error)
    if pg_code == "23514":  # check_violation
        return exc.DBCheckViolationError("Database check constraint violated.", original_exception=sa_error)

    # Class 08: Connection Exception
    if pg_code in ["08000", "08003", "08006"]:
        return exc.DBUnreachableError("Database connection lost.", original_exception=sa_error)
    if pg_code == "08001":
        return exc.DBUnreachableError("Unable to establish connection.", original_exception=sa_error)

    # Class 28: Invalid Authorization
    if pg_code in ["28P01", "28000"]:
        return exc.DBAuthenticationError("Invalid Database Credentials.", original_exception=sa_error)

    # Class 42: Syntax Error or Access Rule Violation (Schema issues)
    if pg_code == "42P01":  # undefined_table
        return exc.DBSchemaError("Table does not exist. Did you run migrations?", original_exception=sa_error)

    # Class 22: Data Exception
    if pg_code == "22P02":  # invalid_text_representation
        return exc.DBInvalidTextRepresentation("Invalid data format (e.g. text in int column).",
                                               original_exception=sa_error)

    # 3. Fallback: Map by Python Exception Type if code is missing
    if isinstance(sa_error, OperationalError):
        return exc.DBUnreachableError("Operational DB Error (Connection/Timeout).", original_exception=sa_error)

    if isinstance(sa_error, ProgrammingError):
        return exc.DBSchemaError("SQL Syntax or Schema Error.", original_exception=sa_error)

    # 4. Catch-all
    return exc.DatabaseError(f"Unknown Database Error: {error_msg}", original_exception=sa_error)