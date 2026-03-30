# Canvas: visual_cue_interfaces_v1
# Path: app/features/visual_cue_detection/domain/interfaces.py

from abc import ABC, abstractmethod

from app.core.model_manager.domain.entities import LoadedModel
from app.features.contextual_retrieval.domain.entities import ContextualizedChunk


class IVisualCueDetectorAdapter(ABC):
    """
    Contract for the NLP model (e.g., Llama 3) that detects visual dependencies.
    """

    @abstractmethod
    def detect_cues(
        self,
        model: LoadedModel,
        chunk: ContextualizedChunk
    ) -> ContextualizedChunk:
        """
        Evaluates a text chunk and returns an updated copy of the chunk
        with the `needs_visual` boolean flag properly set based on SLM analysis.
        """
        pass