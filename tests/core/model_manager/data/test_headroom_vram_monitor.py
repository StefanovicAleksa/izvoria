# Path: tests/core/model_manager/data/test_headroom_vram_monitor.py

from unittest.mock import MagicMock

import pytest

from app.core.model_manager.data.headroom_vram_monitor import HeadroomVramMonitor
from app.core.model_manager.domain.entities import VramStatus


class TestHeadroomVramMonitor:
    @pytest.fixture
    def inner(self):
        inner = MagicMock()
        inner.get_vram_status.return_value = VramStatus(total_vram_mb=24000.0, used_vram_mb=13000.0,
                                                        free_vram_mb=11000.0)
        inner.get_pytorch_memory_stats.return_value = (1.0, 2.0)
        return inner

    def test_free_vram_is_reduced_by_the_headroom(self, inner):
        status = HeadroomVramMonitor(inner, 1500.0).get_vram_status()

        assert status == VramStatus(total_vram_mb=24000.0, used_vram_mb=14500.0, free_vram_mb=9500.0)

    def test_free_vram_never_goes_negative(self, inner):
        assert HeadroomVramMonitor(inner, 50000.0).get_vram_status().free_vram_mb == 0.0

    def test_negative_headroom_is_refused(self, inner):
        with pytest.raises(ValueError):
            HeadroomVramMonitor(inner, -1.0)

    def test_everything_else_is_delegated(self, inner):
        monitor = HeadroomVramMonitor(inner, 100.0)
        monitor.flush_memory()
        monitor.reset_peak_stats()

        assert monitor.get_pytorch_memory_stats() == (1.0, 2.0)
        inner.flush_memory.assert_called_once_with()
        inner.reset_peak_stats.assert_called_once_with()
