# Canvas: model_interfaces_v2
# Path: app/core/model_manager/domain/interfaces.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

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

    @abstractmethod
    def get_pytorch_memory_stats(self) -> Tuple[float, float]:
        """
        Returns PyTorch's internal tensor tracking in MB.
        Format: (current_allocated_mb, peak_allocated_mb)
        """
        pass

    @abstractmethod
    def reset_peak_stats(self) -> None:
        """Resets the peak high-water mark for future tracking."""
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

class IChatModel(ABC):
    """
    A chat-tuned GGUF model as the Qwen adapters use it: the subset of llama_cpp.Llama they call, so an in-process
    Llama and a llama-server client are interchangeable as `LoadedModel.model_instance`. `chat()` returns the
    OpenAI-shaped completion dict (`choices[0].message.content`, `choices[0].finish_reason`).
    """

    @abstractmethod
    def tokenize(self, text: bytes, add_bos: bool = True, special: bool = False) -> List[int]:
        pass

    @abstractmethod
    def n_ctx(self) -> int:
        pass

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], enable_thinking: bool, **kwargs: Any) -> Dict[str, Any]:
        pass

    @abstractmethod
    def close(self) -> None:
        pass
