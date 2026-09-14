# Canvas: test_async_service_v1
# Path: tests/core/model_manager/service/test_async_model_service.py

import pytest
import concurrent.futures
from unittest.mock import MagicMock
from app.core.model_manager.service.async_model_service import AsyncModelService
from app.core.model_manager import exceptions as exc


class TestAsyncModelService:
    def test_resolve_preload_maps_thread_exceptions(self):
        mock_worker = MagicMock()
        service = AsyncModelService(mock_worker)

        # Create a future that raises a raw PyTorch exception
        future = concurrent.futures.Future()
        future.set_exception(RuntimeError("CUDA out of memory"))

        with pytest.raises(exc.OutOfVramError):
            service.resolve_preload(future)

    def test_discard_preload_delegates_to_worker(self):
        mock_worker = MagicMock()
        service = AsyncModelService(mock_worker)
        future, loader = concurrent.futures.Future(), MagicMock()

        service.discard_preload(future, loader)

        mock_worker.discard_preload.assert_called_once_with(future, loader)

    def test_release_all_delegates_and_maps_errors(self):
        mock_worker = MagicMock()
        service = AsyncModelService(mock_worker)

        service.release_all()
        mock_worker.release_all.assert_called_once_with()

        mock_worker.release_all.side_effect = RuntimeError("CUDA out of memory")
        with pytest.raises(exc.OutOfVramError):
            service.release_all()
