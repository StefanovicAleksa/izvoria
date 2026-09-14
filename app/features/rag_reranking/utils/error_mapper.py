# Path: app/features/rag_reranking/utils/error_mapper.py

import logging

from app.core.model_manager.exceptions.base import ModelManagerError
from app.features.rag_reranking import exceptions as exc

logger = logging.getLogger(__name__)


def map_rerank_error(e: Exception, context: str = "") -> exc.RerankingError:
    """
    Translates model-manager, file-system and PyTorch failures into typed reranking exceptions.
    """
    if isinstance(e, exc.RerankingError):
        return e

    error_msg = str(e).lower()

    # 1. Weights missing or unloadable
    if isinstance(e, (FileNotFoundError, ModelManagerError)) or "load" in error_msg and "model" in error_msg:
        return exc.RerankerModelLoadError(
            f"Failed to load the cross-encoder reranker: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 2. Inference failures (OOM, tensor/shape errors)
    if "out of memory" in error_msg or "cuda" in error_msg or "tensor" in error_msg:
        return exc.RerankInferenceError(
            f"Cross-encoder scoring failed: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 3. Fallback
    logger.error(f"Unhandled reranking error: {str(e)}", exc_info=True)
    return exc.RerankingError(
        f"Unexpected failure in RAG reranking: {str(e)} | Context: {context}",
        original_exception=e
    )
