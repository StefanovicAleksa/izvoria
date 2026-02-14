from abc import ABC, abstractmethod
from typing import List, Optional
import uuid

from app.features.storage.domain.entities import Source

class IStorageRepository(ABC):
    @abstractmethod
    def create(self, source: Source) -> Source:
        pass

    @abstractmethod
    def get_by_id(self, source_id: uuid.UUID) -> Optional[Source]:
        pass

    @abstractmethod
    def get_by_hash(self, source_id: str) -> Optional[Source]:
        pass

    @abstractmethod
    def list_all(self) -> List[Source]:
        pass

    @abstractmethod
    def mark_as_processed(self, source_id: uuid.UUID) -> Optional[Source]:
        pass