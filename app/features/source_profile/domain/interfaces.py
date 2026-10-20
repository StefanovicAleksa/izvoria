# Path: app/features/source_profile/domain/interfaces.py

import uuid
from abc import ABC, abstractmethod
from typing import List, Optional

from app.core.model_manager.domain.entities import LoadedModel
from app.features.source_profile.domain.entities import (
    ProfileDraft, ProfiledSource, SourceIdentity, SourceProfile
)


class ISourceProfileRepository(ABC):
    """Contract for persisting source profiles to PostgreSQL, beside the sources they describe."""

    @abstractmethod
    def get_identity(self, source_id: uuid.UUID) -> Optional[SourceIdentity]:
        """The source's name, file name and language, or None if the source does not exist."""
        pass

    @abstractmethod
    def get(self, source_id: uuid.UUID) -> Optional[SourceProfile]:
        """The stored profile of one source, or None."""
        pass

    @abstractmethod
    def save(self, profile: SourceProfile) -> SourceProfile:
        """Inserts or replaces a source's profile."""
        pass

    @abstractmethod
    def list_indexed(self) -> List[ProfiledSource]:
        """Every source that has indexed chunks, with its profile when it has one."""
        pass

    @abstractmethod
    def list_unprofiled(self) -> List[SourceIdentity]:
        """Indexed sources that have no profile yet (the backfill's work list)."""
        pass


class IProfileLanguageAdapter(ABC):
    """
    Contract for the prompts this feature runs on a model somebody else loaded: the contextual SLM during ingestion,
    the answer model at question time. It never loads or releases a model itself.
    """

    @abstractmethod
    def write_profile(self, model: LoadedModel, identity: SourceIdentity, material: str,
                      from_notes: bool = False) -> ProfileDraft:
        """
        A title and topic for a source, from its whole text (or, with `from_notes`, from one note per section of a
        source too long to read at once). Raises ProfileParseError on a reply that is not the expected JSON.
        """
        pass

    @abstractmethod
    def note_section(self, model: LoadedModel, section_text: str) -> str:
        """One or two sentences on what one section of a long source covers, and any title or number it states."""
        pass

    @abstractmethod
    def pick_scope(self, model: LoadedModel, question: str, candidates: List[ProfiledSource]) -> Optional[int]:
        """
        The index in `candidates` of the source the question explicitly names, or None when it names none of them
        (a question about a topic is not a reference to a source).
        """
        pass
