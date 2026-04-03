import gc
import torch
import logging
from typing import Tuple

from app.core.model_manager.domain.interfaces import IVramMonitor
from app.core.model_manager.domain.entities import VramStatus
from app.core.model_manager.utils.error_mapper import map_model_manager_error

logger = logging.getLogger(__name__)


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
                raise RuntimeError("Hardware Error: CUDA device is not available.")

            free_bytes, total_bytes = torch.cuda.mem_get_info()
            used_bytes = total_bytes - free_bytes

            return VramStatus(
                total_vram_mb=total_bytes / (1024 ** 2),
                used_vram_mb=used_bytes / (1024 ** 2),
                free_vram_mb=free_bytes / (1024 ** 2)
            )
        except Exception as e:
            raise map_model_manager_error(e) from e

    def get_pytorch_memory_stats(self) -> Tuple[float, float]:
        """
        Queries PyTorch specifically for Tensor allocations to calculate the "Backpack".
        """
        try:
            if not torch.cuda.is_available():
                return 0.0, 0.0

            current_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
            peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
            return current_alloc, peak_alloc
        except Exception:
            return 0.0, 0.0

    def reset_peak_stats(self) -> None:
        """Resets the peak tracker so we can measure the next phase cleanly."""
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    def flush_memory(self) -> None:
        """
        Aggressively forces garbage collection and clears PyTorch's CUDA cache.
        Includes synchronization to ensure hardware stability during backend swaps.
        """
        try:
            # 1. Drop Python references
            gc.collect()

            if torch.cuda.is_available():
                # 2. Wait for all pending GPU kernels to complete
                torch.cuda.synchronize()

                # 3. Clear the internal PyTorch memory pool
                torch.cuda.empty_cache()

                # 4. Release IPC handles
                torch.cuda.ipc_collect()

            # 5. Final pass for circular references
            gc.collect()
            logger.debug("Aggressive VRAM deep flush completed.")

        except Exception as e:
            raise map_model_manager_error(e) from e