# Canvas: qa_generation_interfaces_v1
# Path: app/features/qa_generation/domain/interfaces.py

from abc import ABC, abstractmethod

from app.core.model_manager.domain.entities import LoadedModel, ModelManifest
from app.core.model_manager.domain.interfaces import IModelLoader


class IQAGenerationAdapter(IModelLoader, ABC):
    """
    Contract for the Generation LLM (e.g., Gemma 4 E4B).
    Inherits from IModelLoader so the Orchestrator manages its VRAM lifecycle.
    """

    @abstractmethod
    def load(self, manifest: ModelManifest) -> LoadedModel:
        """Loads the generation LLM into VRAM."""
        pass

    @abstractmethod
    def unload(self, model: LoadedModel) -> bool:
        """Safely removes the LLM from VRAM and triggers garbage collection."""
        pass

    @abstractmethod
    def generate_answer(
        self,
        model: LoadedModel,
        system_prompt: str,
        context_block: str,
        user_query: str
    ) -> str:
        """
        Executes the inference using the assembled strict context prompt.
        Returns the raw generated string containing citation tags.
        """
        pass