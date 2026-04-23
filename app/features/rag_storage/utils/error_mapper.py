# Canvas: rag_storage_error_mapper_v1
# Path: app/features/rag_storage/utils/error_mapper.py

import logging
from sqlalchemy.exc import SQLAlchemyError
from app.features.rag_storage.exceptions import database as db_exc
from app.features.rag_storage.exceptions import model as mod_exc

logger = logging.getLogger(__name__)


def map_storage_error(e: Exception, context: str = "") -> Exception:
    """
    Translates low-level infrastructure errors into Domain-specific exceptions.
    """
    error_msg = str(e).lower()

    # 1. Handle SQL / pgvector Errors
    if isinstance(e, SQLAlchemyError):
        if "vector" in error_msg and "dimension" in error_msg:
            return db_exc.VectorDatabaseError(
                f"Dimension mismatch in pgvector. Check model output vs DB schema. {context}"
            )
        return db_exc.VectorDatabaseError(f"Database operation failed: {str(e)}. {context}")

    # 2. Handle Embedding / Model Errors
    if "cuda" in error_msg or "vram" in error_msg:
        # We rely on the core ModelManager to catch major OOM,
        # but locally we catch inference specific stalls.
        return mod_exc.EmbeddingInferenceError(f"GPU failure during embedding: {str(e)}")

    if "not loaded" in error_msg:
        return mod_exc.EmbeddingModelNotLoadedError("Embedding model is not in VRAM.")

    # 3. Handle BM25 / File System Errors
    if isinstance(e, (IOError, OSError, PermissionError)):
        return db_exc.SparseIndexError(f"Failed to write BM25 index to disk: {str(e)}")

    # 4. Fallback
    logger.error(f"Unhandled storage error: {str(e)}", exc_info=True)
    return e