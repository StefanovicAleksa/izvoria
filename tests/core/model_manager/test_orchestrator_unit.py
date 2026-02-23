# File: tests/core/model_manager/test_orchestrator_unit.py
import pytest
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
        """Scenario: Plenty of VRAM available, loads safely."""
        # Setup Monitor: 4000MB free (plenty for 1000MB + 500MB buffer)
        mock_monitor.get_vram_status.return_value = VramStatus(
            total_vram_mb=12288.0, used_vram_mb=8288.0, free_vram_mb=4000.0
        )

        # Setup Loader
        fake_loaded = LoadedModel(manifest=sample_manifest, model_instance="fake_pytorch_obj")
        mock_loader.load.return_value = fake_loaded

        # Execute
        result = orchestrator.acquire_model(sample_manifest, mock_loader)

        # Assertions
        assert result == fake_loaded
        assert result.id in orchestrator._active_models
        mock_loader.load.assert_called_once_with(sample_manifest)

    def test_acquire_model_out_of_vram(self, orchestrator, mock_monitor, mock_loader, sample_manifest):
        """Scenario: Not enough VRAM. Bouncer rejects it BEFORE calling load()."""
        # Monitor: Only 1200MB free.
        # Manifest needs 1000MB, but Orchestrator requires +500MB buffer (1500MB total needed).
        mock_monitor.get_vram_status.return_value = VramStatus(
            total_vram_mb=12288.0, used_vram_mb=11088.0, free_vram_mb=1200.0
        )

        with pytest.raises(exc.OutOfVramError, match="Requires 1500.00 MB"):
            orchestrator.acquire_model(sample_manifest, mock_loader)

        # CRITICAL: Ensure the underlying framework loader was never called
        mock_loader.load.assert_not_called()

    def test_acquire_model_idempotent(self, orchestrator, mock_monitor, mock_loader, sample_manifest):
        """Scenario: Requesting a model that is already loaded."""
        # Fake it already being in the dictionary
        existing_model = LoadedModel(manifest=sample_manifest, model_instance="existing_obj")
        orchestrator._active_models[existing_model.id] = existing_model

        # Execute
        result = orchestrator.acquire_model(sample_manifest, mock_loader)

        # Assertions
        assert result == existing_model
        mock_monitor.get_vram_status.assert_not_called()
        mock_loader.load.assert_not_called()

    def test_release_model_success(self, orchestrator, mock_monitor, mock_loader, sample_manifest):
        """Scenario: Safely releasing an active model."""
        active_model = LoadedModel(manifest=sample_manifest, model_instance="obj")
        orchestrator._active_models[active_model.id] = active_model

        orchestrator.release_model(active_model, mock_loader)

        # Assertions
        assert active_model.id not in orchestrator._active_models
        mock_loader.unload.assert_called_once_with(active_model)
        mock_monitor.flush_memory.assert_called_once()

    def test_release_all_emergency_flush(self, orchestrator, mock_monitor):
        """Scenario: Unloading multiple items at once."""
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

        # Execute
        orchestrator.release_all({model_a.id: loader_a, model_b.id: loader_b})

        # Assertions
        assert len(orchestrator._active_models) == 0
        loader_a.unload.assert_called_once_with(model_a)
        loader_b.unload.assert_called_once_with(model_b)
        mock_monitor.flush_memory.assert_called_once()