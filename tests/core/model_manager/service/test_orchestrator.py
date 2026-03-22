# Canvas: test_orchestrator_v1
# Path: tests/core/model_manager/service/test_orchestrator.py

import pytest
from unittest.mock import MagicMock, patch

from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel, VramStatus
from app.core.model_manager.domain.enums import ModelType
from app.core.model_manager import exceptions as exc


class TestModelManagerOrchestrator:
    @pytest.fixture
    def mock_monitor(self):
        monitor = MagicMock()
        # Assume 12GB total, 10GB free by default
        monitor.get_vram_status.return_value = VramStatus(
            total_vram_mb=12000.0, used_vram_mb=2000.0, free_vram_mb=10000.0
        )
        monitor.get_pytorch_memory_stats.return_value = (0.0, 0.0)
        return monitor

    @pytest.fixture
    def mock_loader(self):
        loader = MagicMock()
        return loader

    @pytest.fixture
    def sample_manifest(self):
        return ModelManifest(
            model_type=ModelType.VAD,
            model_path="models/dummy_model.bin",
            required_vram_mb=1000.0
        )

    def test_acquire_model_success_and_idempotent(self, mock_monitor, mock_loader, sample_manifest):
        orchestrator = ModelManagerOrchestrator(mock_monitor)

        # Mock loader to return a dummy LoadedModel
        dummy_loaded = LoadedModel(manifest=sample_manifest, model_instance="fake_tensor")
        mock_loader.load.return_value = dummy_loaded

        # 1. First Load
        result1 = orchestrator.acquire_model(sample_manifest, mock_loader)
        assert result1 == dummy_loaded
        mock_loader.load.assert_called_once()
        mock_monitor.reset_peak_stats.assert_called_once()

        # 2. Second Load (Idempotency)
        result2 = orchestrator.acquire_model(sample_manifest, mock_loader)
        assert result2 == dummy_loaded
        assert mock_loader.load.call_count == 1  # Should NOT call load again

    def test_acquire_model_blocks_oom(self, mock_monitor, mock_loader, sample_manifest):
        orchestrator = ModelManagerOrchestrator(mock_monitor)

        # Mock VRAM to only have 100MB free
        mock_monitor.get_vram_status.return_value = VramStatus(
            total_vram_mb=12000.0, used_vram_mb=11900.0, free_vram_mb=100.0
        )

        with pytest.raises(exc.OutOfVramError) as e_info:
            orchestrator.acquire_model(sample_manifest, mock_loader)

        assert "Cannot load vad" in str(e_info.value)
        assert "but only 100.00 MB is free" in str(e_info.value)
        mock_loader.load.assert_not_called()

    @patch("app.core.model_manager.service.orchestrator.Path")
    def test_get_actual_size_mb_dynamic_discovery(self, mock_path, mock_monitor):
        """Proves the orchestrator checks the OS for real file sizes."""
        orchestrator = ModelManagerOrchestrator(mock_monitor)

        # Setup mock file size to 2048 MB
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.is_file.return_value = True
        mock_path_instance.stat.return_value.st_size = 2048 * (1024 ** 2)
        mock_path.return_value = mock_path_instance

        size = orchestrator._get_actual_size_mb("models/dummy.gguf", 1000.0)
        assert size == 2048.0

    def test_record_telemetry_learns_backpack_spike(self, mock_monitor, mock_loader, sample_manifest):
        orchestrator = ModelManagerOrchestrator(mock_monitor)

        # Load the model
        dummy_loaded = LoadedModel(manifest=sample_manifest, model_instance="fake_tensor")
        mock_loader.load.return_value = dummy_loaded
        orchestrator.acquire_model(sample_manifest, mock_loader)

        # Simulate a telemetry spike (Current: 1000MB, Peak: 2500MB -> Backpack: 1500MB)
        mock_monitor.get_pytorch_memory_stats.return_value = (1000.0, 2500.0)

        orchestrator.record_telemetry()

        # The orchestrator should have saved the 1500MB backpack for this path
        assert orchestrator._historical_backpacks[sample_manifest.model_path] == 1500.0

    def test_release_all_emergency_flush(self, mock_monitor, mock_loader, sample_manifest):
        orchestrator = ModelManagerOrchestrator(mock_monitor)

        dummy_loaded = LoadedModel(manifest=sample_manifest, model_instance="fake")
        mock_loader.load.return_value = dummy_loaded
        orchestrator.acquire_model(sample_manifest, mock_loader)

        loaders_map = {dummy_loaded.id: mock_loader}
        orchestrator.release_all(loaders_map)

        mock_loader.unload.assert_called_once_with(dummy_loaded)
        mock_monitor.flush_memory.assert_called_once()