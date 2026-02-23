# File: tests/core/model_manager/test_vram_monitor.py
import pytest
from unittest.mock import patch
from app.core.model_manager.data.vram_monitor import PyTorchVramMonitor
from app.core.model_manager.domain.entities import VramStatus
from app.core.model_manager import exceptions as exc


class TestPyTorchVramMonitor:

    @pytest.fixture
    def monitor(self):
        return PyTorchVramMonitor()

    @patch("app.core.model_manager.data.vram_monitor.torch.cuda")
    def test_get_vram_status_success(self, mock_cuda, monitor):
        """Verifies correct MB calculation from raw bytes."""
        mock_cuda.is_available.return_value = True

        # mem_get_info returns (free_bytes, total_bytes)
        # Let's mock 4GB free, 12GB total
        free_bytes = 4 * (1024 ** 3)
        total_bytes = 12 * (1024 ** 3)
        mock_cuda.mem_get_info.return_value = (free_bytes, total_bytes)

        status = monitor.get_vram_status()

        assert isinstance(status, VramStatus)
        assert status.total_vram_mb == 12288.0  # 12 * 1024
        assert status.free_vram_mb == 4096.0  # 4 * 1024
        assert status.used_vram_mb == 8192.0  # 8 * 1024

    @patch("app.core.model_manager.data.vram_monitor.torch.cuda")
    def test_get_vram_status_no_cuda(self, mock_cuda, monitor):
        """Verifies failure if hardware doesn't support CUDA."""
        mock_cuda.is_available.return_value = False

        with pytest.raises(exc.ModelManagerError, match="CUDA device is not available"):
            monitor.get_vram_status()

    @patch("app.core.model_manager.data.vram_monitor.gc")
    @patch("app.core.model_manager.data.vram_monitor.torch.cuda")
    def test_flush_memory_success(self, mock_cuda, mock_gc, monitor):
        """Verifies aggressive garbage collection commands are sent."""
        mock_cuda.is_available.return_value = True

        monitor.flush_memory()

        mock_gc.collect.assert_called_once()
        mock_cuda.empty_cache.assert_called_once()
        mock_cuda.ipc_collect.assert_called_once()

    @patch("app.core.model_manager.data.vram_monitor.gc")
    def test_flush_memory_error_mapping(self, mock_gc, monitor):
        """Verifies system crashes during flush are mapped to Domain errors."""
        mock_gc.collect.side_effect = Exception("OS Level GC Crash")

        with pytest.raises(exc.ModelManagerError, match="OS Level GC Crash"):
            monitor.flush_memory()