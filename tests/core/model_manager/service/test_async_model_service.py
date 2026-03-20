# filename: test_async_model_service_unit_v1.py
# path: tests/core/model_manager/test_async_model_service_unit.py
import pytest
import concurrent.futures
from unittest.mock import Mock

from app.core.model_manager.service.async_model_service import AsyncModelService
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.core.model_manager import exceptions as exc


class TestAsyncModelServiceUnit:
    @pytest.fixture
    def mock_worker(self):
        return Mock()

    @pytest.fixture
    def service(self, mock_worker):
        return AsyncModelService(worker=mock_worker)

    @pytest.fixture
    def sample_manifest(self):
        return ModelManifest(model_type=ModelType.VAD, model_path="dummy", required_vram_mb=100)

    def test_trigger_preload_success(self, service, mock_worker, sample_manifest):
        mock_future = concurrent.futures.Future()
        mock_worker.preload_model.return_value = mock_future
        mock_loader = Mock()

        result = service.trigger_preload(sample_manifest, mock_loader)

        assert result == mock_future
        mock_worker.preload_model.assert_called_once_with(sample_manifest, mock_loader)

    def test_resolve_preload_success(self, service):
        """Tests that a successful future returns the LoadedModel."""
        mock_future = concurrent.futures.Future()
        fake_loaded = LoadedModel(
            manifest=ModelManifest(model_type=ModelType.VAD, model_path="D", required_vram_mb=10),
            model_instance="fake"
        )
        mock_future.set_result(fake_loaded)

        result = service.resolve_preload(mock_future)
        assert result == fake_loaded

    def test_resolve_preload_maps_errors(self, service):
        """Tests that if the background thread crashed with a MemoryError, it gets translated."""
        mock_future = concurrent.futures.Future()
        # Simulate a PyTorch memory crash on the background thread
        mock_future.set_exception(MemoryError("Cannot allocate memory"))

        with pytest.raises(exc.OutOfVramError, match="Hardware constraint reached"):
            service.resolve_preload(mock_future)