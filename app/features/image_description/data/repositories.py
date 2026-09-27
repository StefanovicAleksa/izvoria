# Path: app/features/image_description/data/repositories.py

import uuid
from typing import Optional

from sqlalchemy import select

from app.core.database.service.manager import DatabaseManager
from app.features.image_description.data.models import ImageDescriptionModel
from app.features.image_description.domain.entities import ImageDescription
from app.features.image_description.domain.interfaces import IImageDescriptionRepository
from app.features.image_description.utils.error_mapper import map_image_description_error

_FIELDS = ("id", "source_id", "file_name", "shows", "marks", "ocr_text", "ocr_lines_detected", "ocr_lines_read",
           "ocr_withheld_lines", "ocr_checked", "verified_items", "unverified_items", "dropped_items", "taken_at",
           "width", "height", "created_at")


class ImageDescriptionRepository(IImageDescriptionRepository):
    def __init__(self, db_manager: DatabaseManager):
        self._db = db_manager

    def save(self, description: ImageDescription) -> ImageDescription:
        try:
            with self._db.session() as session:
                session.add(ImageDescriptionModel(**{field: getattr(description, field) for field in _FIELDS}))
            return description
        except Exception as e:
            raise map_image_description_error(
                e, context=f"Saving image description for source {description.source_id}") from e

    def get_by_source_id(self, source_id: uuid.UUID) -> Optional[ImageDescription]:
        try:
            with self._db.session() as session:
                row = session.execute(
                    select(ImageDescriptionModel).where(ImageDescriptionModel.source_id == source_id)
                ).scalar_one_or_none()
                if row is None:
                    return None
                return ImageDescription(**{field: getattr(row, field) for field in _FIELDS})
        except Exception as e:
            raise map_image_description_error(e, context=f"Fetching image description for source {source_id}") from e
