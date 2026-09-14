# Path: app/features/rag_reranking/exceptions/model.py

from .base import RerankingError


class RerankerModelLoadError(RerankingError):
    """Raised when the cross-encoder weights cannot be found or loaded into VRAM."""
    pass


class RerankInferenceError(RerankingError):
    """Raised when the cross-encoder fails while scoring query/chunk pairs."""
    pass
