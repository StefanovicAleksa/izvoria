# Canvas: frame_extraction_repositories_v1
# Path: app/features/frame_extraction/data/repositories.py

import uuid
from typing import Optional

from sqlalchemy import select

from app.core.database.service.manager import DatabaseManager
from app.features.frame_extraction.domain.interfaces import IFrameExtractionRepository
from app.features.frame_extraction.domain.entities import ExtractedChunkFrames, ExtractedFrame
from app.features.frame_extraction.data.models import ExtractedChunkFramesModel
from app.features.frame_extraction.utils.error_mapper import map_extraction_error


class FrameExtractionRepository(IFrameExtractionRepository):
    """
    PostgreSQL implementation of the Frame Extraction Repository.
    Handles the serialization of complex Pydantic aggregates into JSONB.
    """

    def __init__(self, db_manager: DatabaseManager):
        self._db = db_manager

    def save(self, chunk_frames: ExtractedChunkFrames) -> ExtractedChunkFrames:
        try:
            # Safely serialize the Pydantic models for Postgres JSONB insertion
            serialized_frames = [frame.model_dump(mode='json') for frame in chunk_frames.frames]

            db_model = ExtractedChunkFramesModel(
                id=chunk_frames.id,
                source_id=chunk_frames.source_id,
                chunk_id=chunk_frames.chunk_id,
                frames=serialized_frames,
                created_at=chunk_frames.created_at
            )

            with self._db.session() as session:
                session.add(db_model)

            return chunk_frames

        except Exception as e:
            # Context string includes "chunk" to trigger specific database mapping
            raise map_extraction_error(e, context=f"FrameExtractionRepository.save for chunk {chunk_frames.chunk_id}") from e

    def get_by_chunk_id(self, chunk_id: uuid.UUID) -> Optional[ExtractedChunkFrames]:
        try:
            with self._db.session() as session:
                stmt = select(ExtractedChunkFramesModel).where(
                    ExtractedChunkFramesModel.chunk_id == chunk_id
                )
                db_model = session.execute(stmt).scalar_one_or_none()

                if not db_model:
                    return None

                # Safely deserialize the JSONB back into strong Pydantic Domain Entities
                domain_frames = [ExtractedFrame(**f) for f in db_model.frames]

                return ExtractedChunkFrames(
                    id=db_model.id,
                    source_id=db_model.source_id,
                    chunk_id=db_model.chunk_id,
                    frames=domain_frames,
                    created_at=db_model.created_at
                )

        except Exception as e:
            raise map_extraction_error(e, context=f"FrameExtractionRepository.get_by_chunk_id {chunk_id}") from e