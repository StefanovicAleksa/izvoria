# Path: app/features/image_description/utils/error_mapper.py

from PIL import Image, UnidentifiedImageError
from sqlalchemy.exc import SQLAlchemyError

from app.core.database import exceptions as core_db_exc
from app.core.model_manager import exceptions as core_model_exc
from app.features.image_description import exceptions as exc
from app.features.visual_analysis.exceptions import ImageDecodeError


def map_image_description_error(e: Exception, context: str = "") -> exc.ImageDescriptionError:
    """
    Translates PIL, filesystem, database and vision errors into this feature's types. Model manager errors
    (OutOfVramError, ...) are re-raised unchanged: the phase and the engine handle those themselves.
    """
    if isinstance(e, exc.ImageDescriptionError):
        return e

    if isinstance(e, core_model_exc.ModelManagerError):
        raise e

    if isinstance(e, (SQLAlchemyError, core_db_exc.DatabaseError)):
        return exc.ImageDescriptionPersistenceError(
            f"Database operation failed in Image Description: {e} | Context: {context}", original_exception=e
        )

    if isinstance(e, FileNotFoundError):
        return exc.SourceImageMissingError(f"Image file not found: {e} | Context: {context}", original_exception=e)

    if isinstance(e, (UnidentifiedImageError, Image.DecompressionBombError, ImageDecodeError)):
        return exc.ImageReadError(f"The image could not be decoded: {e} | Context: {context}", original_exception=e)

    return exc.ImageDescriptionError(
        f"Unexpected failure in Image Description: {e} | Context: {context}", original_exception=e
    )
