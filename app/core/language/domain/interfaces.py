# Path: app/core/language/domain/interfaces.py

from abc import ABC, abstractmethod
from typing import List

from app.core.language.domain.entities import LanguagePack


class ILanguagePackRepository(ABC):
    """Contract for reading language packs from wherever they are kept."""

    @abstractmethod
    def load(self, code: str) -> LanguagePack:
        """The validated pack for a language code. Raises LanguagePackNotFoundError / InvalidLanguagePackError."""
        pass

    @abstractmethod
    def available_codes(self) -> List[str]:
        """Every language code a pack exists for, sorted."""
        pass
