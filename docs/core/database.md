# database

`app/core/database/`

## What it does

Provides the single shared SQLAlchemy engine, session factory, and declarative ORM `Base` that every feature's repository builds on, plus transactional session handling and DB-error translation. This is the one place in the codebase that knows how to talk to Postgres.

## How it does it

`SQLAlchemyEngine` (`data/engine.py`) wraps `create_engine(settings.DATABASE_URL, pool_pre_ping=True, pool_size=settings.DB_POOL_SIZE, max_overflow=settings.DB_MAX_OVERFLOW)` and a `sessionmaker(autocommit=False, autoflush=False)`. `Base(DeclarativeBase)` is also defined here: every feature's ORM model (e.g. `VectorChunkModel`, `File`, `Source`) inherits from it, and the migrations diff against it.

`DatabaseManager` (`service/manager.py`, implementing `IDatabaseManager`) exposes a `@contextmanager session()` that yields a `Session`: commits on success, rolls back and calls `map_database_error(e)` on any `SQLAlchemyError`, and wraps anything else as a generic system error. A module-level singleton, `db_manager = DatabaseManager()`, is instantiated at import time and imported directly by most repositories, `bootstrap.py`, and `OnyxEngine` — repositories are not constructor-injected with a session factory, they import the singleton.

## Migrations

The schema is owned by Alembic. `OnyxEngine.start()` calls `upgrade_to_head(db_manager.get_engine())` (`service/migrations.py`) before anything else touches the database:

1. A database with Onyx tables but no `alembic_version` predates migrations. The statements in `data/migrations/pre_alembic.py` (frozen, never edited) run for the tables it has. Then revision `0001` runs, which creates only the tables and indexes still missing, in the baseline's shape. Its data is kept. Don't bring back `create_all()` here: it builds a missing table from today's models, and the later revision that adds one of its columns then fails.
2. `alembic upgrade head` runs in the same transaction. Postgres DDL is transactional, so a failed revision leaves the database untouched and raises `DBMigrationError`, which stops the boot. A database that is down or refuses the credentials still raises `DBUnreachableError`/`DBAuthenticationError`.

Revisions live in `data/migrations/versions/`. `env.py` imports every `app/**/data/models.py` (`utils/model_registry.import_all_models`) so autogenerate always sees the whole schema.

**Adding a column or table:**

1. Change the model.
2. Generate against a **fresh** database brought to head, never `db_test` itself. Integration tests drop foreign keys there, and autogenerate would write them back into your revision:
   ```
   docker compose exec -T db_test psql -U postgres -c "DROP DATABASE IF EXISTS onyx_alembic_scratch" -c "CREATE DATABASE onyx_alembic_scratch"
   ENV=test POSTGRES_DB=onyx_alembic_scratch venv/bin/alembic upgrade head
   ENV=test POSTGRES_DB=onyx_alembic_scratch venv/bin/alembic revision --autogenerate --rev-id 000N -m "add foo"
   ```
3. Read the revision. Autogenerate renders `Computed` expressions and server defaults as literals: `fts_document` must keep `settings.FTS_LANGUAGE_CONFIG`. It also can't see renames (it writes a drop plus an add, which loses the data). Write a backfill for any new `NOT NULL` column.
4. `integration_tests/core/database/test_migrations_integration.py` fails while models and revisions disagree.

Migrations are forward-only: `downgrade()` raises. The way back from a bad update is a database backup.

## Inputs / Outputs

Input: nothing beyond `app/core/config.py`'s `DATABASE_URL`/pool settings. Output: a context-managed `Session` per `db_manager.session()` call, used by every feature's `data/` repository layer for reads/writes.

## Where it fits

Not part of the pipeline itself — it's cross-cutting infrastructure that every repository across every feature depends on. `OnyxEngine.start()`/`stop()` own its lifecycle (`upgrade_to_head()` on start, `db_manager.dispose()` on stop).

## Gotchas / design notes

- Switching `APPLIANCE_LANGUAGE` still needs a reset: `fts_document` is generated with the old stemmer, and no revision regenerates it.

- The singleton pattern means the engine connects eagerly at import time, not lazily on first use.
- `dispose()` must be called explicitly on shutdown (`OnyxEngine.stop()` does this) or pooled connections leak past process exit.
- All repositories share one connection pool sized by `DB_POOL_SIZE`/`DB_MAX_OVERFLOW` — a feature that holds a session open across a long GPU operation would starve the pool for everyone else; repositories should acquire/release sessions tightly around the actual query.
