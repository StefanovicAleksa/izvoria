import pytest
from unittest.mock import patch, MagicMock
from app.core.model_manager.data.vram_monitor import PyTorchVramMonitor
from app.core.model_manager.domain.entities import VramStatus
from app.core.model_manager import exceptions as exc


class TestPyTorchVramMonitor:
    @patch("app.core.model_manager.data.vram_monitor.torch")
    def test_get_vram_status_success(self, mock_torch):
        mock_torch.cuda.is_available.return_value = True
        # 8GB free, 12GB total
        mock_torch.cuda.mem_get_info.return_value = (8 * 1024 ** 2, 12 * 1024 ** 2)

        monitor = PyTorchVramMonitor()
        status = monitor.get_vram_status()

        assert isinstance(status, VramStatus)
        assert status.total_vram_mb == 12.0
        assert status.free_vram_mb == 8.0
        assert status.used_vram_mb == 4.0

    @patch("app.core.model_manager.data.vram_monitor.torch")
    def test_get_vram_status_fails_if_cuda_unavailable(self, mock_torch):
        mock_torch.cuda.is_available.return_value = False

        monitor = PyTorchVramMonitor()
        with pytest.raises(exc.ModelManagerError) as e_info:
            monitor.get_vram_status()
        assert "CUDA device is not available" in str(e_info.value)

    @patch("app.core.model_manager.data.vram_monitor.torch")
    def test_get_pytorch_memory_stats_success(self, mock_torch):
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 1 * 1024 ** 2
        mock_torch.cuda.max_memory_allocated.return_value = 3 * 1024 ** 2

        monitor = PyTorchVramMonitor()
        current, peak = monitor.get_pytorch_memory_stats()

        assert current == 1.0
        assert peak == 3.0

    @patch("app.core.model_manager.data.vram_monitor.torch")
    def test_flush_memory_success(self, mock_torch):
        """Verifies the deep flush sequence including hardware synchronization."""
        mock_torch.cuda.is_available.return_value = True

        monitor = PyTorchVramMonitor()
        monitor.flush_memory()

        # FIXED: Ensure hardware sync is called to prevent race conditions
        mock_torch.cuda.synchronize.assert_called_once()
        mock_torch.cuda.empty_cache.assert_called_once()
        mock_torch.cuda.ipc_collect.assert_called_once()