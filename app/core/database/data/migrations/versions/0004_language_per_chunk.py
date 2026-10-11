"""language per chunk: one index holds every language, each chunk keyword-indexed with its own stemmer

Until now the stemmer was written into the schema: `fts_document` was generated as
to_tsvector('<the appliance's language>', text_payload), so a database served exactly one language. Each chunk
now carries its own text search configuration (`fts_config`) and `fts_document` is generated from it. Existing
chunks are stamped with the configuration the old column was built with, read back from the column itself.

`sources.language` and `research_reports.language` record which language a file was ingested in and a report was
written in. They are backfilled with the code of the language the database was built in: the configured one when
its stemmer matches the index, otherwise from the stemmer. `simple` maps to Slovenian, the only pack using it
today; if another pack ever uses `simple` too, that mapping is ambiguous, and such rows are left for the operator.

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-29 19:30:37.773075
"""
import re

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from app.core.config import settings

revision = '0004'
down_revision = '0003'
branch_labels = None
depends_on = None

# The generated column's own expression is the record of which stemmer built the index
_FTS_EXPRESSION_SQL = sa.text("""
    SELECT pg_get_expr(d.adbin, d.adrelid) AS expression
    FROM pg_attrdef d
    JOIN pg_attribute a ON a.attrelid = d.adrelid AND a.attnum = d.adnum
    WHERE d.adrelid = to_regclass('vector_chunks') AND a.attname = 'fts_document'
""")
_REGCONFIG = re.compile(r"'([a-z_]+)'::regconfig")

# Frozen with this revision: the stemmer each shipped pack used when it was written
_LANGUAGE_BY_STEMMER = {"english": "en", "serbian": "sr", "simple": "sl"}


def _indexed_stemmer(connection) -> str:
    expression = connection.execute(_FTS_EXPRESSION_SQL).scalar()
    match = _REGCONFIG.search(expression or "")
    return match.group(1) if match else settings.FTS_LANGUAGE_CONFIG


def _language_code(stemmer: str):
    if stemmer == settings.FTS_LANGUAGE_CONFIG:
        return settings.APPLIANCE_LANGUAGE
    return _LANGUAGE_BY_STEMMER.get(stemmer)


def upgrade() -> None:
    connection = op.get_bind()
    stemmer = _indexed_stemmer(connection)
    language = _language_code(stemmer)

    op.add_column('vector_chunks', sa.Column('fts_config', postgresql.REGCONFIG(), nullable=True))
    connection.execute(sa.text("UPDATE vector_chunks SET fts_config = CAST(:stemmer AS regconfig)"),
                       {"stemmer": stemmer})
    op.alter_column('vector_chunks', 'fts_config', nullable=False)

    # A generated column's expression can't be altered: drop it and add it back over the new column. Postgres
    # recomputes every row. to_tsvector(regconfig, text) is immutable, which a generated column requires.
    op.drop_index('ix_vector_chunks_fts', table_name='vector_chunks')
    op.drop_column('vector_chunks', 'fts_document')
    op.add_column('vector_chunks', sa.Column(
        'fts_document', postgresql.TSVECTOR(),
        sa.Computed("to_tsvector(fts_config, text_payload)", persisted=True), nullable=True
    ))
    op.create_index('ix_vector_chunks_fts', 'vector_chunks', ['fts_document'], unique=False,
                    postgresql_using='gin')

    op.add_column('sources', sa.Column('language', sa.String(), nullable=True))
    op.add_column('research_reports', sa.Column('language', sa.String(), nullable=True))
    if language:
        connection.execute(sa.text("UPDATE sources SET language = :language"), {"language": language})
        connection.execute(sa.text("UPDATE research_reports SET language = :language"), {"language": language})


def downgrade() -> None:
    # The appliance only moves forward; a restore from backup is the way back.
    raise NotImplementedError("Izvoria migrations are forward-only.")
