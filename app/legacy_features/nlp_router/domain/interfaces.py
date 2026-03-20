# Canvas: nlp_router_interfaces_v1
# Path: app/features/nlp_router/domain/interfaces.py

from abc import ABC, abstractmethod

from app.core.model_manager.domain.interfaces import IModelLoader
from app.core.model_manager.domain.entities import LoadedModel

from app.legacy_features.nlp_router.domain.entities import TranscriptChunk, RouterReasoning


class ILLMRouterAdapter(IModelLoader, ABC):
    """
    Contract for the local LLM inference engine (e.g., llama.cpp running Phi-3.5).
    Guarantees it can both manage VRAM lifecycle and extract visual cues from text.
    """

    @abstractmethod
    def extract_visual_cues(
        self,
        model: LoadedModel,
        chunk: TranscriptChunk
    ) -> RouterReasoning:
        """
        Processes a block of transcript text and returns the strictly-typed
        reasoning and time windows extracted by the LLM.

        Args:
            model: The VRAM-loaded instance of the LLM.
            chunk: The chronological block of transcript text to analyze.

        Returns:
            RouterReasoning: The parsed and validated JSON response.
        """
        pass