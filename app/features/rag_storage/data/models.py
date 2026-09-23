# Canvas: rag_storage_models_v3
# Path: app/features/rag_storage/data/models.py

from sqlalchemy import Column, String, Float, Integer, ForeignKey, Index, Computed
from sqlalchemy.dialects.postgresql import UUID, TSVECTOR, JSONB
from pgvector.sqlalchemy import Vector
import uuid

from app.core.config import settings
from app.core.database.data.engine import Base


class VectorChunkModel(Base):
    """
    SQLAlchemy Model for Hybrid RAG Storage.
    Stores dense mathematical embeddings alongside Postgres TSVECTORs for sparse FTS.
    """
    __tablename__ = "vector_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_id = Column(UUID(as_uuid=True), ForeignKey("sources.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_id = Column(UUID(as_uuid=True), nullable=False, unique=True, index=True)

    # Where the chunk sits in its source; `anchor_kind` says which of the fields below mean something.
    # start_sec/end_sec are 0.0 for document chunks, page_* are NULL for transcripts.
    anchor_kind = Column(String, nullable=False, default="time", server_default="time")
    start_sec = Column(Float, nullable=False)
    end_sec = Column(Float, nullable=False)
    page_start = Column(Integer, nullable=True)
    page_end = Column(Integer, nullable=True)
    # Heading path of a document chunk ("Article 7 Termination › 7.2"); NULL for transcripts
    section_path = Column(String, nullable=True)

    # Reading order within the source. Neighbours and reading order use this rather than start_sec, which is
    # meaningless for documents.
    position = Column(Integer, nullable=True)

    # Contextualized text (generated context + transcript): the embedded / keyword-indexed payload
    text_payload = Column(String, nullable=False)

    # Verbatim speaker-labelled transcript text: the only text handed to the answer model
    raw_text = Column(String, nullable=False)

    # Generated audio/visual context, kept separately so it is never presented as citable source
    context_text = Column(String, nullable=True)

    speaker_labels = Column(JSONB, nullable=False, default=list)

    embedding = Column(Vector(1024), nullable=False)

    # Automatically tokenizes the text_payload using the configured Postgres text search dictionary
    fts_document = Column(
        TSVECTOR,
        Computed(f"to_tsvector('{settings.FTS_LANGUAGE_CONFIG}', text_payload)", persisted=True)
    )

    __table_args__ = (
        Index('ix_vector_chunks_fts', 'fts_document', postgresql_using='gin'),
        Index('ix_vector_chunks_source_position', 'source_id', 'position'),
    )

    def __repr__(self):
        return f"<VectorChunkModel(chunk_id={self.chunk_id}, source_id={self.source_id})>"


# There are no migrations: create_all() never alters a table that already exists. OnyxEngine.start() runs these
# after create_all(). Every statement must be idempotent -- this runs on every boot.
VECTOR_CHUNKS_UPGRADE_DDL: tuple = (
    # Added with document ingestion. Chunks indexed before it are transcripts: time-anchored, and read in
    # start_sec order, which is what the position backfill reproduces.
    "ALTER TABLE vector_chunks ADD COLUMN IF NOT EXISTS anchor_kind TEXT NOT NULL DEFAULT 'time'",
    "ALTER TABLE vector_chunks ADD COLUMN IF NOT EXISTS page_start INTEGER",
    "ALTER TABLE vector_chunks ADD COLUMN IF NOT EXISTS page_end INTEGER",
    "ALTER TABLE vector_chunks ADD COLUMN IF NOT EXISTS section_path TEXT",
    "ALTER TABLE vector_chunks ADD COLUMN IF NOT EXISTS position INTEGER",
    # Only rows from before the column have no position, and this runs on every boot: the partial index stays
    # empty once they are backfilled, so the check below costs nothing instead of sorting the whole table
    "CREATE INDEX IF NOT EXISTS ix_vector_chunks_position_missing ON vector_chunks (source_id) WHERE position IS NULL",
    """UPDATE vector_chunks v SET position = ordered.rn - 1
       FROM (SELECT id, ROW_NUMBER() OVER (PARTITION BY source_id ORDER BY start_sec, end_sec) AS rn
             FROM vector_chunks
             WHERE source_id IN (SELECT source_id FROM vector_chunks WHERE position IS NULL)) ordered
       WHERE v.id = ordered.id AND v.position IS NULL""",
    "CREATE INDEX IF NOT EXISTS ix_vector_chunks_source_position ON vector_chunks (source_id, position)",
)
