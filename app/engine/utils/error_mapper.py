# Canvas: engine_error_mapper_v1
# Path: app/engine/utils/error_mapper.py

import logging

from app.engine import exceptions as engine_exc
from app.orchestration.exceptions.base import OrchestrationError
from app.features.rag_retrieval.exceptions.base import RetrievalError
from app.core.database.exceptions.base import DatabaseError
from app.core.model_manager.exceptions.base import ModelManagerError

logger = logging.getLogger(__name__)


def map_engine_error(e: Exception, context: str = "") -> engine_exc.EngineError:
    """
    Translates underlying Orchestration, Feature, and Core Infrastructure
    errors into unified Engine exceptions.
    """
    # Guard against double wrapping
    if isinstance(e, engine_exc.EngineError):
        return e

    # 1. Appliance Hardware / Infrastructure Errors (Startup issues)
    if isinstance(e, (DatabaseError, ModelManagerError)):
        return engine_exc.EngineBootError(
            f"Appliance infrastructure failure: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 2. Ingestion / Pipeline Errors
    if isinstance(e, OrchestrationError):
        return engine_exc.EngineError(
            f"Appliance media ingestion failed: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 3. RAG Search Errors
    if isinstance(e, RetrievalError):
        return engine_exc.EngineError(
            f"Appliance search query failed: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 4. Fallback Catch-all
    logger.error(f"Unhandled Engine Error: {str(e)}", exc_info=True)
    return engine_exc.EngineError(
        f"An unexpected appliance error occurred: {str(e)} | Context: {context}",
        original_exception=e
    )