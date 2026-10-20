# Path: app/orchestration/service/phases/source_scope_phase.py

import logging
import uuid
from typing import List, Optional

from app.core.config import settings
from app.features.qa_generation.domain.entities import QAResponse
from app.features.qa_generation.utils.context_packer import chunk_source_text
from app.features.rag_retrieval.domain.entities import RetrievedChunk
from app.features.source_profile.domain.entities import SourceScope
from app.features.source_profile.service.source_profile_api import SourceProfileAPI

logger = logging.getLogger(__name__)

# Characters per token, on the low side for every served language, so a source judged to fit really does
_CHARS_PER_TOKEN = 3.0


class SourceScopePhase:
    """
    Tells which source a question names ("what are all the formulas in lecture 4 of Stat 110?"), so the live
    pipelines answer from that source alone. Holds no model and no preload: resolving borrows the answer model
    synchronously, so pipelines call it before any preload of their own (as they do the follow-up rewrite).
    """

    def __init__(self, profile_api: SourceProfileAPI):
        self._profile_api = profile_api

    def resolve(self, search_id: uuid.UUID, question: str) -> Optional[SourceScope]:
        """The source the question names, or None. A failure only costs the scoping, so it searches everything."""
        try:
            return self._profile_api.resolve_scope(question)
        except Exception as e:
            logger.error(f"Resolving the source named in Search {search_id} failed; searching every source: {e}")
            return None

    @staticmethod
    def fits_whole(chunks: List[RetrievedChunk]) -> bool:
        """
        Whether a source is short enough to hand the answer model whole (a lecture is ~13K tokens). A whole source
        is what answers "list every …" about it; one that doesn't fit is searched within instead.
        """
        if not chunks:
            return False
        estimated_tokens = sum(len(chunk_source_text(chunk)) for chunk in chunks) / _CHARS_PER_TOKEN
        return estimated_tokens <= settings.SOURCE_SCOPE_WHOLE_READ_MAX_TOKENS

    @staticmethod
    def mark(response: Optional[QAResponse], scope: Optional[SourceScope]) -> Optional[QAResponse]:
        """The answer, marked with the source it was read from, so a wrongly picked source is visible."""
        if response is None or scope is None:
            return response
        return response.model_copy(update={"scoped_source_title": scope.title})
