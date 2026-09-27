# Path: app/features/image_description/exceptions/database.py

from .base import ImageDescriptionError


class ImageDescriptionPersistenceError(ImageDescriptionError):
    """An image description could not be saved to or read from PostgreSQL."""
    pass
