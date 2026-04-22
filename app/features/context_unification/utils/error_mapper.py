# Canvas: context_unification_error_mapper_v1
# Path: app/features/context_unification/utils/error_mapper.py

from sqlalchemy.exc import SQLAlchemyError
from pydantic import ValidationError

from app.core.database import exceptions as core_db_exc
from app.features.context_unification import exceptions as exc


def map_unification_error(e: Exception, context: str = "") -> exc.ContextUnificationError:
    """
    Translates raw system and Database exceptions into
    highly specific typed Domain Exceptions for the Context Unification feature.
    """
    if isinstance(e, exc.ContextUnificationError):
        return e

    # 1. Database Errors
    if isinstance(e, (SQLAlchemyError, core_db_exc.DatabaseError)):
        if "transcript" in context.lower() or "save" in context.lower():
            return exc.UnifiedTranscriptPersistenceError(
                f"Failed to serialize and save Unified Transcript to PostgreSQL: {str(e)}",
                original_exception=e
            )
        return exc.ContextUnificationDatabaseError(
            f"Database operation failed in Context Unification: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 2. Data Validation Errors
    if isinstance(e, ValidationError):
        return exc.ContextUnificationPipelineError(
            f"Pydantic validation failed while assembling Unified chunks: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 3. Fallback Catch-all
    return exc.ContextUnificationError(
        f"Unexpected failure in Context Unification: {str(e)} | Context: {context}",
        original_exception=e
    )