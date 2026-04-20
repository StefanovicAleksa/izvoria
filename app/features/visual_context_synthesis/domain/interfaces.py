# Canvas: visual_synthesis_interfaces_v2
# Path: app/features/visual_context_synthesis/domain/interfaces.py

import uuid
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any

from app.core.model_manager.domain.interfaces import IModelLoader
from app.core.model_manager.domain.entities import LoadedModel

from app.features.visual_context_synthesis.domain.entities import VisualTimeline


class IVisualSLMAdapter(IModelLoader, ABC):
    """
    Contract for the SLM Adapter that will synthesize the visual timeline into a fluid paragraph.
    Inherits from IModelLoader to support strict VRAM management via the Orchestrator.
    """

    @abstractmethod
    def synthesize_context(
        self,
        model: LoadedModel,
        visual_events: List[Dict[str, Any]]
    ) -> str:
        """
        Takes the chronological, fuzzy-grouped list of visual events and
        prompts the SLM to write a cohesive 1-2 sentence visual summary.
        """
        pass


class IVisualSynthesisRepository(ABC):
    """
    Contract for persisting the VisualTimeline to PostgreSQL.
    """

    @abstractmethod
    def save(self, timeline: VisualTimeline) -> VisualTimeline:
        """Saves a newly synthesized visual timeline to the database."""
        pass

    @abstractmethod
    def get_by_source_id(self, source_id: uuid.UUID) -> Optional[VisualTimeline]:
        """Retrieves a previously computed visual timeline for a given Source."""
        pass