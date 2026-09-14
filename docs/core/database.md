# database

`app/core/database/`

## What it does

Provides the single shared SQLAlchemy engine, session factory, and declarative ORM `Base` that every feature's repository builds on, plus transactional session handling and DB-error translation. This is the one place in the codebase that knows how to talk to Postgres.

## How it does it

`SQLAlchemyEngine` (`data/engine.py`) wraps `create_engine(settings.DATABASE_URL, pool_pre_ping=True, pool_size=settings.DB_POOL_SIZE, max_overflow=settings.DB_MAX_OVERFLOW)` and a `sessionmaker(autocommit=False, autoflush=False)`. `Base(DeclarativeBase)` is also defined here and is what `OnyxEngine.start()` calls `Base.metadata.create_all()` against, and what every feature's ORM model (e.g. `VectorChunkModel`, `File`, `Source`) inherits from.

`DatabaseManager` (`service/manager.py`, implementing `IDatabaseManager`) exposes a `@contextmanager session()` that yields a `Session`: commits on success, rolls back and calls `map_database_error(e)` on any `SQLAlchemyError`, and wraps anything else as a generic system error. A module-level singleton, `db_manager = DatabaseManager()`, is instantiated at import time and imported directly by most repositories, `bootstrap.py`, and `OnyxEngine` — repositories are not constructor-injected with a session factory, they import the singleton.

## Inputs / Outputs

Input: nothing beyond `app/core/config.py`'s `DATABASE_URL`/pool settings. Output: a context-managed `Session` per `db_manager.session()` call, used by every feature's `data/` repository layer for reads/writes.

## Where it fits

Not part of the pipeline itself — it's cross-cutting infrastructure that every repository across every feature depends on. `OnyxEngine.start()`/`stop()` own its lifecycle (`create_all()` on start, `db_manager.dispose()` on stop).

## Gotchas / design notes

- The singleton pattern means the engine connects eagerly at import time, not lazily on first use.
- `dispose()` must be called explicitly on shutdown (`OnyxEngine.stop()` does this) or pooled connections leak past process exit.
- All repositories share one connection pool sized by `DB_POOL_SIZE`/`DB_MAX_OVERFLOW` — a feature that holds a session open across a long GPU operation would starve the pool for everyone else; repositories should acquire/release sessions tightly around the actual query.
