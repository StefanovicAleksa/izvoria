# Path: integration_tests/core/database/test_migrations_integration.py

import uuid

import psycopg2
import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url

from app.core.config import settings
from app.core.database.data.engine import Base
from alembic.script import ScriptDirectory

from app.core.database.service.migrations import MIGRATIONS_DIR, _config, upgrade_to_head

pytestmark = pytest.mark.integration

HEAD = ScriptDirectory(str(MIGRATIONS_DIR)).get_current_head()

SCRATCH_DB = "onyx_test_migrations"

# Columns and tables that earlier builds did not have, removed again to fake a database from before migrations
_PRE_ALEMBIC_DROPS = (
    "ALTER TABLE vector_chunks DROP COLUMN anchor_kind, DROP COLUMN page_start, DROP COLUMN page_end, "
    "DROP COLUMN section_path, DROP COLUMN position",
    "ALTER TABLE session_turns DROP COLUMN mode, DROP COLUMN standalone_query, DROP COLUMN evidence_chunk_ids, "
    "DROP COLUMN first_doc_number",
    "ALTER TABLE research_reports DROP COLUMN mode, DROP COLUMN rounds_used",
    "ALTER TABLE research_report_evidence DROP COLUMN anchor_kind, DROP COLUMN page_start, DROP COLUMN page_end, "
    "DROP COLUMN section_path",
    "DROP TABLE image_descriptions",
    "DROP TABLE alembic_version",
)


def _admin(sql: str) -> None:
    connection = psycopg2.connect(host=settings.POSTGRES_SERVER, port=settings.POSTGRES_PORT,
                                  user=settings.POSTGRES_USER, password=settings.POSTGRES_PASSWORD,
                                  dbname=settings.POSTGRES_DB)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql)
    finally:
        connection.close()


def _drift(engine) -> list:
    with engine.connect() as connection:
        context = MigrationContext.configure(connection, opts={
            "compare_type": True,
            "include_object": lambda obj, name, type_, reflected, compare_to:
                not (type_ == "table" and name == "alembic_version"),
        })
        return compare_metadata(context, Base.metadata)


def _revision(engine) -> str:
    with engine.connect() as connection:
        return connection.execute(text("SELECT version_num FROM alembic_version")).scalar()


@pytest.fixture
def scratch_engine():
    """An empty database of its own, so the shared test schema is never touched."""
    _admin(f'DROP DATABASE IF EXISTS "{SCRATCH_DB}" WITH (FORCE)')
    _admin(f'CREATE DATABASE "{SCRATCH_DB}"')
    engine = create_engine(make_url(settings.DATABASE_URL).set(database=SCRATCH_DB))
    try:
        yield engine
    finally:
        engine.dispose()
        _admin(f'DROP DATABASE IF EXISTS "{SCRATCH_DB}" WITH (FORCE)')


class TestMigrationsIntegration:
    def test_empty_database_reaches_head_with_no_drift_from_the_models(self, scratch_engine):
        upgrade_to_head(scratch_engine)

        assert _revision(scratch_engine) == HEAD
        # Fails when a model changes without a revision: generate one with `alembic revision --autogenerate`
        assert _drift(scratch_engine) == []

    def test_upgrading_twice_is_a_no_op(self, scratch_engine):
        upgrade_to_head(scratch_engine)
        upgrade_to_head(scratch_engine)

        assert _revision(scratch_engine) == HEAD

    def test_baseline_stems_keyword_search_in_the_appliance_language(self, scratch_engine):
        upgrade_to_head(scratch_engine)

        with scratch_engine.connect() as connection:
            expression = connection.execute(text(
                "SELECT pg_get_expr(d.adbin, d.adrelid) FROM pg_attrdef d JOIN pg_attribute a "
                "ON a.attrelid = d.adrelid AND a.attnum = d.adnum "
                "WHERE d.adrelid = 'vector_chunks'::regclass AND a.attname = 'fts_document'")).scalar()

        assert f"'{settings.FTS_LANGUAGE_CONFIG}'::regconfig" in expression

    def test_pre_alembic_database_is_caught_up_and_stamped_with_its_data_kept(self, scratch_engine):
        upgrade_to_head(scratch_engine)
        source_id, file_id, turn_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        with scratch_engine.begin() as connection:
            for statement in _PRE_ALEMBIC_DROPS:
                connection.execute(text(statement))
            connection.execute(text(
                "INSERT INTO files (id, file_path, file_name, file_size_bytes, is_processed, created_at) "
                "VALUES (:id, '/v/a.wav', 'a.wav', 1, true, now())"), {"id": file_id})
            connection.execute(text(
                "INSERT INTO sources (id, name, source_type, content_hash, file_id, created_at, updated_at) "
                "VALUES (:id, 'a.wav', 'audio', 'h', :fid, now(), now())"), {"id": source_id, "fid": file_id})
            for start in (30.0, 0.0):
                connection.execute(text(
                    "INSERT INTO vector_chunks (id, source_id, chunk_id, start_sec, end_sec, text_payload, raw_text, "
                    "speaker_labels, embedding) VALUES (gen_random_uuid(), :sid, gen_random_uuid(), :s, :e, 't', "
                    "'t', '[]', :emb)"), {"sid": source_id, "s": start, "e": start + 30, "emb": str([0.01] * 1024)})
            connection.execute(text(
                "INSERT INTO session_turns (id, session_id, question, answer_text, created_at) "
                "VALUES (:id, gen_random_uuid(), 'q', 'a', now())"), {"id": turn_id})
        assert not inspect(scratch_engine).has_table("alembic_version")

        upgrade_to_head(scratch_engine)

        assert _revision(scratch_engine) == HEAD
        assert _drift(scratch_engine) == []
        with scratch_engine.connect() as connection:
            chunks = connection.execute(text(
                "SELECT start_sec, position, anchor_kind FROM vector_chunks ORDER BY start_sec")).all()
            turn_mode = connection.execute(text("SELECT mode FROM session_turns WHERE id = :id"),
                                           {"id": turn_id}).scalar()
        assert [tuple(c) for c in chunks] == [(0.0, 0, "time"), (30.0, 1, "time")]
        assert turn_mode == "chat"

    def test_downgrade_is_refused(self, scratch_engine):
        upgrade_to_head(scratch_engine)

        with scratch_engine.begin() as connection:
            with pytest.raises(NotImplementedError, match="forward-only"):
                command.downgrade(_config(connection), "base")
