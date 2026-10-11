# Path: app/core/database/data/migrations/env.py
#
# Run two ways: by upgrade_to_head() on every IzvoriaEngine.start(), which hands over its own connection through
# config.attributes, and by a developer's `alembic` command (alembic.ini at the repo root), which connects to
# settings.DATABASE_URL -- so ENV decides which database a new revision is generated against.

from alembic import context
from sqlalchemy import create_engine

from app.core.config import settings
from app.core.database.data.engine import Base
from app.core.database.utils.model_registry import import_all_models

import_all_models()
target_metadata = Base.metadata


def _include_object(obj, name, type_, reflected, compare_to) -> bool:
    # alembic_version is Alembic's own bookkeeping, not a model
    return not (type_ == "table" and name == "alembic_version")


def _run(connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        include_object=_include_object,
        compare_type=True,
        transaction_per_migration=False,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connection = context.config.attributes.get("connection")
    if connection is not None:
        _run(connection)
        return

    engine = create_engine(settings.DATABASE_URL)
    try:
        with engine.begin() as connection:
            _run(connection)
    finally:
        engine.dispose()


if context.is_offline_mode():
    raise RuntimeError("Offline (--sql) migrations are not supported: the baseline reads the appliance's settings.")

run_migrations_online()
