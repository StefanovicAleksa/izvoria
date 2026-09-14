# Canvas: rag_storage_error_mapper_v2
# Path: app/features/rag_storage/utils/error_mapper.py

import logging
from sqlalchemy.exc import SQLAlchemyError

from app.core.database import exceptions as core_db_exc
from app.core.model_manager.exceptions.base import ModelManagerError
from app.features.rag_storage.exceptions.base import RAGStorageError
from app.features.rag_storage.exceptions import database as db_exc
from app.features.rag_storage.exceptions import model as mod_exc

logger = logging.getLogger(__name__)


def map_storage_error(e: Exception, context: str = "") -> RAGStorageError:
    """
    Translates low-level infrastructure errors into Domain-specific exceptions.
    Always returns a RAGStorageError so callers up the stack only see feature-level errors.
    """
    if isinstance(e, RAGStorageError):
        return e

    error_msg = str(e).lower()

    # 1. Handle SQL / pgvector Errors (raw SQLAlchemy or already translated by the core DatabaseManager)
    if isinstance(e, (SQLAlchemyError, core_db_exc.DatabaseError)):
        if "vector" in error_msg and "dimension" in error_msg:
            return db_exc.VectorDatabaseError(
                f"Dimension mismatch in pgvector. Check model output vs DB schema. {context}",
                original_exception=e
            )
        return db_exc.VectorDatabaseError(f"Database operation failed: {str(e)}. {context}", original_exception=e)

    # 2. Handle Embedding / Model Errors
    if isinstance(e, ModelManagerError) or "cuda" in error_msg or "vram" in error_msg:
        # We rely on the core ModelManager to catch major OOM,
        # but locally we catch inference specific stalls.
        return mod_exc.EmbeddingInferenceError(f"GPU failure during embedding: {str(e)}. {context}",
                                               original_exception=e)

    if "not loaded" in error_msg:
        return mod_exc.EmbeddingModelNotLoadedError(f"Embedding model is not in VRAM. {context}",
                                                    original_exception=e)

    # 3. Handle File System Errors
    if isinstance(e, (IOError, OSError, PermissionError)):
        return db_exc.SparseIndexError(f"Storage I/O failure: {str(e)}. {context}", original_exception=e)

    # 4. Fallback
    logger.error(f"Unhandled storage error: {str(e)}", exc_info=True)
    return RAGStorageError(f"Unexpected RAG storage failure: {str(e)}. {context}", original_exception=e)
