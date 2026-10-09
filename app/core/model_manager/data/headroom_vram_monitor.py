# File: app/core/model_manager/data/headroom_vram_monitor.py

from typing import Tuple

from app.core.model_manager.domain.entities import VramStatus
from app.core.model_manager.domain.interfaces import IVramMonitor


class HeadroomVramMonitor(IVramMonitor):
    """
    Reports `headroom_mb` less free VRAM than the card has. The hosted ingestion worker shares one card with the
    live API (llama-server plus the pinned embedder and reranker); admitting its loads against the card minus the
    live side's share keeps a batch load from taking memory a chat is about to need. The live processes' usage is
    already subtracted by the driver, so the headroom covers only their growth (KV fill, activation spikes).
    """

    def __init__(self, inner: IVramMonitor, headroom_mb: float):
        if headroom_mb < 0:
            raise ValueError("VRAM headroom cannot be negative")
        self._inner = inner
        self._headroom_mb = headroom_mb

    def get_vram_status(self) -> VramStatus:
        status = self._inner.get_vram_status()
        free = max(0.0, status.free_vram_mb - self._headroom_mb)
        return VramStatus(total_vram_mb=status.total_vram_mb,
                          used_vram_mb=status.total_vram_mb - free,
                          free_vram_mb=free)

    def flush_memory(self) -> None:
        self._inner.flush_memory()

    def get_pytorch_memory_stats(self) -> Tuple[float, float]:
        return self._inner.get_pytorch_memory_stats()

    def reset_peak_stats(self) -> None:
        self._inner.reset_peak_stats()
