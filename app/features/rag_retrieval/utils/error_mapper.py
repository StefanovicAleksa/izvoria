# Canvas: rag_retrieval_error_mapper_v2
# Path: app/features/rag_retrieval/utils/error_mapper.py

import logging
from sqlalchemy.exc import SQLAlchemyError, ProgrammingError

from app.core.model_manager.exceptions.base import ModelManagerError
from app.features.rag_retrieval.exceptions import database as db_exc
from app.features.rag_retrieval.exceptions import model as mod_exc
from app.features.rag_retrieval.exceptions.base import RetrievalError

logger = logging.getLogger(__name__)

def map_retrieval_error(e: Exception, context: str = "") -> RetrievalError:
    """
    Translates low-level system, DB, and Model errors into Domain-specific exceptions.
    """
    if isinstance(e, RetrievalError):
        return e

    error_msg = str(e).lower()

    # 1. Database & SQL Errors
    if isinstance(e, ProgrammingError):
        if "syntax error in tsquery" in error_msg:
            return db_exc.InvalidSearchSyntaxError(
                f"Invalid characters in search prompt. {context}",
                original_exception=e
            )
    if isinstance(e, SQLAlchemyError):
        return db_exc.SearchExecutionError(
            f"Database query failed: {str(e)}. {context}",
            original_exception=e
        )

    # 2. Model & Inference Errors
    if isinstance(e, ModelManagerError):
        return mod_exc.PromptEmbeddingError(
            f"Failed to load or execute embedding model: {str(e)}. {context}",
            original_exception=e
        )
    if "token indices sequence length" in error_msg or "context length" in error_msg:
        return mod_exc.ContextLengthExceededError(
            f"Search prompt is too long for the embedding model. {context}",
            original_exception=e
        )

    # 3. Fallback
    logger.error(f"Unhandled retrieval error: {str(e)}", exc_info=True)
    return RetrievalError(
        f"An unexpected error occurred during retrieval: {str(e)}. {context}",
        original_exception=e
    )