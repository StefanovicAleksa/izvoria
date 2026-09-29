# Canvas: rag_storage_models_v3
# Path: app/features/rag_storage/data/models.py

from sqlalchemy import Column, String, Float, Integer, ForeignKey, Index, Computed
from sqlalchemy.dialects.postgresql import UUID, TSVECTOR, JSONB, REGCONFIG
from sqlalchemy.dialects.postgresql.base import ischema_names
from pgvector.sqlalchemy import Vector
import uuid

from app.core.config import settings
from app.core.database.data.engine import Base

# SQLAlchemy can declare REGCONFIG but does not reflect it; without this the migration drift check sees an
# unknown type on vector_chunks.fts_config.
ischema_names.setdefault("regconfig", REGCONFIG)


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

    # The Postgres text search configuration of the language this chunk was ingested in ("serbian"). One index
    # holds every language, so each chunk carries its own stemmer. REGCONFIG, not text: a generated column needs
    # an immutable expression, and to_tsvector(regconfig, text) is one while a text::regconfig cast is not.
    # Defaults to the active language's, the language a batch is ingested in.
    fts_config = Column(REGCONFIG, nullable=False, default=lambda: settings.FTS_LANGUAGE_CONFIG)

    # Tokenizes text_payload with the chunk's own text search configuration
    fts_document = Column(
        TSVECTOR,
        Computed("to_tsvector(fts_config, text_payload)", persisted=True)
    )

    __table_args__ = (
        Index('ix_vector_chunks_fts', 'fts_document', postgresql_using='gin'),
        Index('ix_vector_chunks_source_position', 'source_id', 'position'),
    )

    def __repr__(self):
        return f"<VectorChunkModel(chunk_id={self.chunk_id}, source_id={self.source_id})>"
