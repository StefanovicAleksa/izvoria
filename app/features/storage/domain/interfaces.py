# filename: storage_interfaces_v4.py
# path: app/features/storage/domain/interfaces.py
from abc import ABC, abstractmethod
from typing import List, Optional
import uuid

from app.features.storage.domain.entities import Source, File

class IStorageRepository(ABC):
    @abstractmethod
    def create(self, source: Source) -> Source:
        """Saves a new Source (and its associated File)."""
        pass

    @abstractmethod
    def create_file(self, file: File) -> File:
        """Saves a standalone physical file record."""
        pass

    @abstractmethod
    def get_by_id(self, source_id: uuid.UUID) -> Optional[Source]:
        """Fetches a full Source aggregate (Source + File)."""
        pass

    @abstractmethod
    def get_file_by_id(self, file_id: uuid.UUID) -> Optional[File]:
        """Fetches a standalone physical file record."""
        pass

    @abstractmethod
    def delete_file(self, file_id: uuid.UUID) -> Optional[File]:
        """Deletes a standalone physical file record. Returns the deleted record, or None if absent."""
        pass

    @abstractmethod
    def get_by_hash(self, content_hash: str) -> Optional[Source]:
        """Fetches a Source by its content hash for deduplication."""
        pass

    @abstractmethod
    def list_all(self) -> List[Source]:
        """Lists all stored sources."""
        pass

    @abstractmethod
    def mark_as_processed(self, source_id: uuid.UUID) -> Optional[Source]:
        """Marks the underlying file of a source as processed."""
        pass

    @abstractmethod
    def set_language(self, source_id: uuid.UUID, language: Optional[str]) -> Source:
        """Records the language a source is ingested in; None un-stamps it."""
        pass
