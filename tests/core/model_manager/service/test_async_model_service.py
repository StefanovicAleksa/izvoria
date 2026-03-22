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