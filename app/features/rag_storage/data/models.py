# Canvas: rag_storage_models_v1
# Path: app/features/rag_storage/data/models.py

from sqlalchemy import Column, String, Float, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
import uuid

from app.core.database.data.engine import Base


class VectorChunkModel(Base):
    """
    SQLAlchemy Model for the Dense Vector Index.
    Stores the mathematical embeddings and the exact provenance metadata.
    """
    __tablename__ = "vector_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Foreign key linking back to the physical file/video in Phase 1
    # Adjust the ForeignKey string if your sources table is named differently.
    source_id = Column(UUID(as_uuid=True), ForeignKey("sources.id", ondelete="CASCADE"), nullable=False, index=True)

    # The exact chunk ID from the UnifiedTranscript
    chunk_id = Column(UUID(as_uuid=True), nullable=False, unique=True, index=True)

    # Strict temporal metadata for UI clip playback
    start_sec = Column(Float, nullable=False)
    end_sec = Column(Float, nullable=False)

    # The full contextualized text used to generate the embedding
    text_payload = Column(String, nullable=False)

    # The actual semantic vector. BGE-M3 outputs 1024 dimensions.
    embedding = Column(Vector(1024), nullable=False)

    def __repr__(self):
        return f"<VectorChunkModel(chunk_id={self.chunk_id}, source_id={self.source_id})>"