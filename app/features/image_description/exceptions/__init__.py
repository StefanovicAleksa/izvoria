# Path: app/features/image_description/exceptions/__init__.py

from .base import ImageDescriptionError
from .image import SourceImageMissingError, ImageReadError, EmptyImageError
from .database import ImageDescriptionPersistenceError

__all__ = [
    "ImageDescriptionError",
    "SourceImageMissingError",
    "ImageReadError",
    "EmptyImageError",
    "ImageDescriptionPersistenceError",
]
