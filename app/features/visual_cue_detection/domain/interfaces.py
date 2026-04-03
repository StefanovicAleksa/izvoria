# app/features/visual_cue_detection/domain/interfaces.py

from abc import ABC, abstractmethod
from app.core.model_manager.domain.interfaces import IModelLoader # Add this
from app.core.model_manager.domain.entities import LoadedModel, ModelManifest
from app.features.contextual_retrieval.domain.entities import ContextualizedChunk

class IVisualCueDetectorAdapter(IModelLoader, ABC): # Inherit from IModelLoader
    """Contract for the NLP model (e.g., Qwen 2.5) that detects visual dependencies."""

    @abstractmethod
    def load(self, manifest: ModelManifest) -> LoadedModel:
        pass

    @abstractmethod
    def unload(self, model: LoadedModel) -> bool: # Change return type to bool
        pass

    @abstractmethod
    def detect_cues(self, model: LoadedModel, chunk: ContextualizedChunk) -> ContextualizedChunk:
        pass