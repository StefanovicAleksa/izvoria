# Path: app/core/database/service/migrations.py

import logging
import threading
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import SQLAlchemyError

from app.core.database import exceptions as exc
from app.core.database.data.migrations.pre_alembic import catch_up_statements
from app.core.database.utils.error_mapper import map_database_error

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "data" / "migrations"
# The schema as it stood when Alembic was introduced; a database from an earlier build is caught up to it
BASELINE_REVISION = "0001"
# Present in every database any Izvoria build has created, and the first table a baseline creates
_SENTINEL_TABLE = "sources"
# Alembic installs the running migration's context into module globals (`alembic.context`, `alembic.op`), so two
# upgrades at once in one process can run one database's revisions on the other's connection. The hosted API starts
# one engine per firm, and two firms' first requests may arrive together: their migrations take turns.
_UPGRADE_LOCK = threading.Lock()


def _config(connection: Connection) -> Config:
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    config.attributes["connection"] = connection
    return config


def _is_pre_alembic(connection: Connection) -> bool:
    """Izvoria tables, but no record of which revision they are at: a database from before migrations existed."""
    inspector = inspect(connection)
    return not inspector.has_table("alembic_version") and inspector.has_table(_SENTINEL_TABLE)


def upgrade_to_head(engine: Engine) -> None:
    """
    Brings the database to the newest schema, in one transaction: Postgres DDL is transactional, so a revision
    that fails leaves the database exactly as it was, and the raised DBMigrationError stops the engine booting.

    A database from before Alembic first gets the frozen pre-Alembic statements for the tables it has, then the
    baseline -- which creates only the tables and indexes it lacks -- then is upgraded like any other. An
    appliance full of ingested data is kept, not reset.
    """
    try:
        with _UPGRADE_LOCK, engine.begin() as connection:
            config = _config(connection)
            if _is_pre_alembic(connection):
                # The tables it has get their missing columns first; then the baseline creates only what is still
                # missing, in its own shape. create_all() built a missing table from today's models instead, and the
                # later revision that adds one of its columns failed.
                logger.info(f"Database predates migrations: catching it up to revision {BASELINE_REVISION}.")
                for statement in catch_up_statements(set(inspect(connection).get_table_names())):
                    connection.execute(text(statement))
                command.upgrade(config, BASELINE_REVISION)
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
