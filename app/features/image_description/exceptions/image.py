# Path: app/features/image_description/exceptions/image.py

from .base import ImageDescriptionError


class SourceImageMissingError(ImageDescriptionError):
    """The source has no file record, or its file is gone from the vault."""
    pass


class ImageReadError(ImageDescriptionError):
    """The file is not an image PIL can decode: corrupt, truncated, an unsupported format, or a decompression bomb."""
    pass


class EmptyImageError(ImageDescriptionError):
    """Neither the vision model nor OCR found anything in the image: there is nothing to index or cite."""
    pass
