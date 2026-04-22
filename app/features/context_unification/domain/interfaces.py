# Canvas: context_unification_interfaces_v1
# Path: app/features/context_unification/domain/interfaces.py

import uuid
from abc import ABC, abstractmethod
from typing import Optional

from app.features.context_unification.domain.entities import UnifiedTranscript


class IContextUnificationRepository(ABC):
    """
    Contract for persisting the finalized, RAG-ready UnifiedTranscript to PostgreSQL.
    """

    @abstractmethod
    def save(self, transcript: UnifiedTranscript) -> UnifiedTranscript:
        """Saves a newly assembled Unified Transcript to the database."""
        pass

    @abstractmethod
    def get_by_source_id(self, source_id: uuid.UUID) -> Optional[UnifiedTranscript]:
        """Retrieves a previously computed Unified Transcript for a given Source."""
        pass