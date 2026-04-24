# Canvas: clip_extractor_repositories_v1
# Path: app/features/clip_extractor/data/repositories.py

import uuid
from typing import List, Optional
from sqlalchemy import select

from app.core.database.domain.interfaces import IDatabaseManager
from app.features.clip_extractor.domain.interfaces import IVideoClipRepository
from app.features.clip_extractor.domain.entities import VideoClip
from app.features.clip_extractor.data.models import VideoClipModel
from app.features.clip_extractor.utils.error_mapper import map_clip_extractor_error


class VideoClipRepository(IVideoClipRepository):
    def __init__(self, db_manager: IDatabaseManager):
        self._db = db_manager

    def create_bulk(self, clips: List[VideoClip]) -> List[VideoClip]:
        if not clips:
            return []

        try:
            with self._db.session() as session:
                models_to_insert = [
                    VideoClipModel(
                        id=clip.id,
                        source_id=clip.source_id,
                        chunk_id=clip.chunk_id,
                        extracted_file_id=clip.extracted_file_id,
                        start_sec=clip.start_sec,
                        end_sec=clip.end_sec
                    )
                    for clip in clips
                ]

                # Bulk save is significantly faster for database inserts
                session.bulk_save_objects(models_to_insert)

            return clips
        except Exception as e:
            raise map_clip_extractor_error(e, context="Saving bulk video clips to database")

    def get_by_chunk_id(self, chunk_id: uuid.UUID) -> Optional[VideoClip]:
        try:
            with self._db.session() as session:
                stmt = select(VideoClipModel).where(VideoClipModel.chunk_id == chunk_id)
                model = session.execute(stmt).scalar_one_or_none()

                if not model:
                    return None

                return VideoClip.model_validate(model)
        except Exception as e:
            raise map_clip_extractor_error(e, context=f"Fetching video clip for chunk {chunk_id}")

    def get_by_source_id(self, source_id: uuid.UUID) -> List[VideoClip]:
        try:
            with self._db.session() as session:
                stmt = select(VideoClipModel).where(VideoClipModel.source_id == source_id)
                models = session.execute(stmt).scalars().all()

                return [VideoClip.model_validate(m) for m in models]
        except Exception as e:
            raise map_clip_extractor_error(e, context=f"Fetching video clips for source {source_id}")