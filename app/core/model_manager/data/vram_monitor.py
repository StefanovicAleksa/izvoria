# File: app/core/model_manager/data/vram_monitor.py
import gc
import torch

from app.core.model_manager.domain.interfaces import IVramMonitor
from app.core.model_manager.domain.entities import VramStatus
from app.core.model_manager.utils.error_mapper import map_model_manager_error


class PyTorchVramMonitor(IVramMonitor):
    """
    Concrete implementation of the VRAM Monitor using PyTorch's CUDA API.
    """

    def get_vram_status(self) -> VramStatus:
        """
        Queries the NVIDIA GPU for absolute memory metrics.
        Returns the status in Megabytes (MB).
        """
        try:
            if not torch.cuda.is_available():
                # For Onyx, we strictly require the provided NVIDIA workstation hardware.
                raise RuntimeError("Hardware Error: CUDA device is not available.")

            # mem_get_info() returns (free_bytes, total_bytes)
            # This is highly accurate as it bypasses the PyTorch allocator caching
            # and queries the underlying NVIDIA driver directly.
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            used_bytes = total_bytes - free_bytes

            return VramStatus(
                total_vram_mb=total_bytes / (1024 ** 2),
                used_vram_mb=used_bytes / (1024 ** 2),
                free_vram_mb=free_bytes / (1024 ** 2)
            )

        except Exception as e:
            raise map_model_manager_error(e) from e

    def flush_memory(self) -> None:
        """
        Aggressively forces garbage collection and clears PyTorch's CUDA cache.
        This is critical before loading a new model to prevent OutOfVram errors.
        """
        try:
            # 1. Force Python to clean up unreferenced objects (like unloaded models)
            gc.collect()

            # 2. Force PyTorch to release its cached memory blocks back to the GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

        except Exception as e:
            raise map_model_manager_error(e) from e