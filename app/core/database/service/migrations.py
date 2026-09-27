# Path: app/core/database/service/migrations.py

import logging
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import SQLAlchemyError

from app.core.database import exceptions as exc
from app.core.database.data.engine import Base
from app.core.database.data.migrations.pre_alembic import PRE_ALEMBIC_CATCH_UP
from app.core.database.utils.error_mapper import map_database_error
from app.core.database.utils.model_registry import import_all_models

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "data" / "migrations"
# The schema as it stood when Alembic was introduced; a database from an earlier build is caught up to it
BASELINE_REVISION = "0001"
# Present in every database any Onyx build has created, and the first table a baseline creates
_SENTINEL_TABLE = "sources"


def _config(connection: Connection) -> Config:
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    config.attributes["connection"] = connection
    return config


def _is_pre_alembic(connection: Connection) -> bool:
    """Onyx tables, but no record of which revision they are at: a database from before migrations existed."""
    inspector = inspect(connection)
    return not inspector.has_table("alembic_version") and inspector.has_table(_SENTINEL_TABLE)


def upgrade_to_head(engine: Engine) -> None:
    """
    Brings the database to the newest schema, in one transaction: Postgres DDL is transactional, so a revision
    that fails leaves the database exactly as it was, and the raised DBMigrationError stops the engine booting.

    A database from before Alembic is first caught up to the baseline with create_all() and the frozen
    pre-Alembic statements, then stamped, so an appliance full of ingested data is kept, not reset.
    """
    try:
        with engine.begin() as connection:
            config = _config(connection)
            if _is_pre_alembic(connection):
                logger.info(f"Database predates migrations: catching it up and stamping revision {BASELINE_REVISION}.")
                import_all_models()
                Base.metadata.create_all(bind=connection)
                for statement in PRE_ALEMBIC_CATCH_UP:
                    connection.execute(text(statement))
                command.stamp(config, BASELINE_REVISION)
            command.upgrade(config, "head")
    except exc.DatabaseError:
        raise
    except SQLAlchemyError as e:
        # A database that is down or refuses the credentials says so, rather than posing as a bad migration
        mapped = map_database_error(e)
        if type(mapped) is exc.DatabaseError or isinstance(mapped, exc.SystemError):
            raise exc.DBMigrationError(f"Database migration failed: {e}", original_exception=e) from e
        raise mapped from e
    except Exception as e:
        raise exc.DBMigrationError(f"Database migration failed: {e}", original_exception=e) from e
