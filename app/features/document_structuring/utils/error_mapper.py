# Path: app/features/document_structuring/utils/error_mapper.py

import re

from app.features.document_structuring import exceptions as exc


def map_structuring_error(e: Exception, context: str = "") -> exc.DocumentStructuringError:
    """Structuring is pure Python; anything it raises is a chunking bug or a bad pack regex."""
    if isinstance(e, exc.DocumentStructuringError):
        return e
    if isinstance(e, (ValueError, IndexError, re.error)):
        return exc.ChunkingError(f"Failed to split the document into chunks: {e} | Context: {context}",
                                 original_exception=e)
    return exc.DocumentStructuringError(f"Unexpected failure in Document Structuring: {e} | Context: {context}",
                                        original_exception=e)
