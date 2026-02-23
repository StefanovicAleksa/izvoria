# File: app/core/model_manager/domain/interfaces.py
from abc import ABC, abstractmethod
from typing import Any, Dict

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel, VramStatus


class IVramMonitor(ABC):
    """Contract for checking and flushing GPU Memory."""

    @abstractmethod
    def get_vram_status(self) -> VramStatus:
        """Queries the hardware for current memory usage."""
        pass

    @abstractmethod
    def flush_memory(self) -> None:
        """Forces Python garbage collection and clears the CUDA cache."""
        pass


class IModelLoader(ABC):
    """
    Contract for adapters that load AI models into VRAM.
    Each supported framework (e.g., NeMoAdapter, CTranslateAdapter) will implement this.
    """

    @abstractmethod
    def load(self, manifest: ModelManifest) -> LoadedModel:
        """
        Loads the model weights from disk into VRAM.
        Should raise a domain exception (e.g., OutOfVramError) if it fails.
        """
        pass

    @abstractmethod
    def unload(self, model: LoadedModel) -> bool:
        """
        Safely deletes the model instance from memory.
        """
        pass