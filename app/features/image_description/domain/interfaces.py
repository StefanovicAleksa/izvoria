# Path: app/features/image_description/domain/interfaces.py

import uuid
from abc import ABC, abstractmethod
from typing import Optional

from app.features.image_description.domain.entities import ImageDescription


class IImageDescriptionRepository(ABC):
    @abstractmethod
    def save(self, description: ImageDescription) -> ImageDescription:
        pass

    @abstractmethod
    def get_by_source_id(self, source_id: uuid.UUID) -> Optional[ImageDescription]:
        pass
