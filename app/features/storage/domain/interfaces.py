# filename: storage_interfaces_v2.py
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
        pass

    @abstractmethod
    def get_by_hash(self, content_hash: str) -> Optional[Source]:
        pass

    @abstractmethod
    def list_all(self) -> List[Source]:
        pass

    @abstractmethod
    def mark_as_processed(self, source_id: uuid.UUID) -> Optional[Source]:
        pass