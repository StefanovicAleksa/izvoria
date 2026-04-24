# Canvas: rag_storage_models_v2
# Path: app/features/rag_storage/data/models.py

from sqlalchemy import Column, String, Float, ForeignKey, Index, Computed
from sqlalchemy.dialects.postgresql import UUID, TSVECTOR
from pgvector.sqlalchemy import Vector
import uuid

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

    start_sec = Column(Float, nullable=False)
    end_sec = Column(Float, nullable=False)

    text_payload = Column(String, nullable=False)
    embedding = Column(Vector(1024), nullable=False)

    # Automatically tokenizes the text_payload using Postgres' English dictionary
    fts_document = Column(TSVECTOR, Computed("to_tsvector('english', text_payload)", persisted=True))

    __table_args__ = (
        Index('ix_vector_chunks_fts', 'fts_document', postgresql_using='gin'),
    )

    def __repr__(self):
        return f"<VectorChunkModel(chunk_id={self.chunk_id}, source_id={self.source_id})>"