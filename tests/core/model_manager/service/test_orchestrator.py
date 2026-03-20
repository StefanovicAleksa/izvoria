# filename: test_orchestrator_unit_v2.py
# path: tests/core/model_manager/test_orchestrator.py
import pytest
import concurrent.futures
from unittest.mock import Mock

from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel, VramStatus
from app.core.model_manager.domain.enums import ModelType
from app.core.model_manager import exceptions as exc


class TestModelManagerOrchestrator:

    @pytest.fixture
    def mock_monitor(self):
        return Mock()

    @pytest.fixture
    def mock_loader(self):
        return Mock()

    @pytest.fixture
    def orchestrator(self, mock_monitor):
        return ModelManagerOrchestrator(vram_monitor=mock_monitor)

    @pytest.fixture
    def sample_manifest(self):
        return ModelManifest(
            model_type=ModelType.VAD,
            model_path="/vault/weights/vad.bin",
            required_vram_mb=1000.0
        )

    def test_acquire_model_success(self, orchestrator, mock_monitor, mock_loader, sample_manifest):
        mock_monitor.get_vram_status.return_value = VramStatus(
            total_vram_mb=12288.0, used_vram_mb=8288.0, free_vram_mb=4000.0
        )
        fake_loaded = LoadedModel(manifest=sample_manifest, model_instance="fake_pytorch_obj")
        mock_loader.load.return_value = fake_loaded

        result = orchestrator.acquire_model(sample_manifest, mock_loader)

        assert result == fake_loaded
        assert result.id in orchestrator._active_models
        mock_loader.load.assert_called_once_with(sample_manifest)

    def test_acquire_model_out_of_vram(self, orchestrator, mock_monitor, mock_loader, sample_manifest):
        mock_monitor.get_vram_status.return_value = VramStatus(
            total_vram_mb=12288.0, used_vram_mb=11088.0, free_vram_mb=1200.0
        )

        with pytest.raises(exc.OutOfVramError, match="Requires 1500.00 MB"):
            orchestrator.acquire_model(sample_manifest, mock_loader)

        mock_loader.load.assert_not_called()

    def test_acquire_model_idempotent(self, orchestrator, mock_monitor, mock_loader, sample_manifest):
        existing_model = LoadedModel(manifest=sample_manifest, model_instance="existing_obj")
        orchestrator._active_models[existing_model.id] = existing_model

        result = orchestrator.acquire_model(sample_manifest, mock_loader)

        assert result == existing_model
        mock_monitor.get_vram_status.assert_not_called()
        mock_loader.load.assert_not_called()

    def test_release_model_success(self, orchestrator, mock_monitor, mock_loader, sample_manifest):
        active_model = LoadedModel(manifest=sample_manifest, model_instance="obj")
        orchestrator._active_models[active_model.id] = active_model

        orchestrator.release_model(active_model, mock_loader)

        assert active_model.id not in orchestrator._active_models
        mock_loader.unload.assert_called_once_with(active_model)
        mock_monitor.flush_memory.assert_called_once()

    def test_release_all_emergency_flush(self, orchestrator, mock_monitor):
        loader_a, loader_b = Mock(), Mock()
        model_a = LoadedModel(
            manifest=ModelManifest(model_type=ModelType.VAD, model_path="A", required_vram_mb=10),
            model_instance="a"
        )
        model_b = LoadedModel(
            manifest=ModelManifest(model_type=ModelType.TRANSCRIPTION, model_path="B", required_vram_mb=10),
            model_instance="b"
        )
        orchestrator._active_models = {model_a.id: model_a, model_b.id: model_b}

        orchestrator.release_all({model_a.id: loader_a, model_b.id: loader_b})

        assert len(orchestrator._active_models) == 0
        loader_a.unload.assert_called_once_with(model_a)
        loader_b.unload.assert_called_once_with(model_b)
        mock_monitor.flush_memory.assert_called_once()

    def test_thread_safety_concurrent_acquires(self, orchestrator, mock_monitor, mock_loader, sample_manifest):
        """
        Scenario: Multiple threads try to load the EXACT SAME model simultaneously.
        The lock should ensure the hardware is only queried once, the model is only loaded once,
        and subsequent threads get the cached instance.
        """
        mock_monitor.get_vram_status.return_value = VramStatus(
            total_vram_mb=12288.0, used_vram_mb=8288.0, free_vram_mb=4000.0
        )
        fake_loaded = LoadedModel(manifest=sample_manifest, model_instance="fake_pytorch_obj")
        mock_loader.load.return_value = fake_loaded

        # Spam the orchestrator with 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(orchestrator.acquire_model, sample_manifest, mock_loader) for _ in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All 10 requests should return the exact same LoadedModel object
        assert all(r == fake_loaded for r in results)
        assert len(orchestrator._active_models) == 1

        # Hardware queries and loads should strictly only happen ONCE
        mock_monitor.get_vram_status.assert_called_once()
        mock_loader.load.assert_called_once()