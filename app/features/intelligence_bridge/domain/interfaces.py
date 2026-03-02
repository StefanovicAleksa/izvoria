import uuid
from abc import ABC, abstractmethod
from typing import Optional

from app.features.intelligence_bridge.domain.entities import IntelligentSceneMap

class IIntelligentSceneRepository(ABC):
    """Contract for persisting the deduplicated, multimodal Scene Map."""

    @abstractmethod
    def save_map(self, scene_map: IntelligentSceneMap) -> IntelligentSceneMap:
        """Persists the final enriched map to the database."""
        pass

    @abstractmethod
    def get_map_by_source_id(self, source_id: uuid.UUID) -> Optional[IntelligentSceneMap]:
        """Retrieves a saved Intelligent Scene Map for a given video source."""
        pass