# Path: app/features/source_profile/utils/error_mapper.py

import json
import logging

from pydantic import ValidationError

from app.core.database import exceptions as db_exc
from app.core.model_manager.exceptions import ModelManagerError
from app.features.source_profile import exceptions as exc

logger = logging.getLogger(__name__)


def map_source_profile_error(e: Exception, context: str = "") -> Exception:
    """
    Translates database, model and parsing exceptions into Source Profile exceptions. A ModelManagerError passes
    through unchanged: phases and the engine catch it to release models.
    """
    if isinstance(e, (exc.SourceProfileError, ModelManagerError)):
        return e

    suffix = f" | Context: {context}" if context else ""

    if isinstance(e, (json.JSONDecodeError, ValidationError)):
        return exc.ProfileParseError(f"The model's profile reply was not usable: {e}{suffix}", original_exception=e)

    if isinstance(e, (db_exc.DataError, db_exc.IntegrityError, ValueError)):
        return exc.SourceProfileDataError(f"Invalid source profile: {e}{suffix}", original_exception=e)

    if isinstance(e, db_exc.DatabaseError):
        return exc.SourceProfileSystemError(
            f"Source profiles are unavailable due to a database failure: {e}{suffix}", original_exception=e
        )

    error_msg = str(e).lower()
    if "llama" in error_msg or "out of memory" in error_msg or "context window" in error_msg:
        return exc.ProfileGenerationError(f"The language model failed: {e}{suffix}", original_exception=e)

    logger.error(f"Unhandled Source Profile error: {e}", exc_info=True)
    return exc.SourceProfileError(f"Unexpected failure in Source Profile: {e}{suffix}", original_exception=e)
