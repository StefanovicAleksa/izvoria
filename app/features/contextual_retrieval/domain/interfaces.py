# Canvas: contextual_interfaces_v2
# Path: app/features/contextual_retrieval/domain/interfaces.py

import uuid
from abc import ABC, abstractmethod
from typing import Optional

from app.core.model_manager.domain.interfaces import IModelLoader
from app.core.model_manager.domain.entities import LoadedModel
from app.features.contextual_retrieval.domain.entities import ContextualizedTranscript


class ISLMContextAdapter(IModelLoader, ABC):
    """
    Contract for the Small Language Model (SLM) Adapter.
    Inherits from IModelLoader to support strict VRAM management via the Orchestrator.
    """

    @abstractmethod
    def generate_context(
        self,
        model: LoadedModel,
        full_document_text: str,
        chunk_text: str
    ) -> str:
        """
        Uses prompt caching to efficiently evaluate the chunk against the full document.
        Returns a concise 50-100 word context string explaining the chunk.
        """
        pass


class IContextualRepository(ABC):
    """
    Contract for persisting the ContextualizedTranscript to PostgreSQL.
    Stores the chunks as highly optimized JSONB.
    """

    @abstractmethod
    def save(self, transcript: ContextualizedTranscript) -> ContextualizedTranscript:
        """Saves a new Contextualized Transcript to the database."""
        pass

    @abstractmethod
    def update(self, transcript: ContextualizedTranscript) -> ContextualizedTranscript:
        """
        Overwrites the JSONB chunks column for an existing transcript.
        Used to persist state changes (like `needs_visual` flags) without replacing the whole row.
        """
        pass

    @abstractmethod
    def get_by_source_id(self, source_id: uuid.UUID) -> Optional[ContextualizedTranscript]:
        """Retrieves a previously computed Contextualized Transcript for a given Source."""
        pass