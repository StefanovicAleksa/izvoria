# filename: test_background_worker_v1.py
# path: tests/core/model_manager/test_background_worker.py
import pytest
import concurrent.futures
from unittest.mock import Mock

from app.core.model_manager.utils.background_worker import BackgroundModelWorker
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType


class TestBackgroundModelWorker:
    @pytest.fixture
    def mock_orchestrator(self):
        return Mock()

    @pytest.fixture
    def worker(self, mock_orchestrator):
        worker = BackgroundModelWorker(orchestrator=mock_orchestrator)
        yield worker
        worker.shutdown(wait=True)

    @pytest.fixture
    def sample_manifest(self):
        return ModelManifest(model_type=ModelType.VAD, model_path="dummy", required_vram_mb=100)

    @pytest.fixture
    def mock_loader(self):
        return Mock()

    def test_preload_model_success(self, worker, mock_orchestrator, sample_manifest, mock_loader):
        """Should submit a task that calls orchestrator.acquire_model and returns the Future."""
        fake_loaded = LoadedModel(manifest=sample_manifest, model_instance="fake_obj")
        mock_orchestrator.acquire_model.return_value = fake_loaded

        future = worker.preload_model(sample_manifest, mock_loader)

        assert isinstance(future, concurrent.futures.Future)
        result = future.result(timeout=2.0)

        assert result == fake_loaded
        mock_orchestrator.acquire_model.assert_called_once_with(sample_manifest, mock_loader)

    def test_unload_model_success(self, worker, mock_orchestrator, sample_manifest, mock_loader):
        """Should submit a task that calls orchestrator.release_model."""
        fake_loaded = LoadedModel(manifest=sample_manifest, model_instance="fake_obj")

        future = worker.unload_model(fake_loaded, mock_loader, flush=True)

        assert isinstance(future, concurrent.futures.Future)
        result = future.result(timeout=2.0)

        assert result is True
        mock_orchestrator.release_model.assert_called_once_with(fake_loaded, mock_loader, flush=True)