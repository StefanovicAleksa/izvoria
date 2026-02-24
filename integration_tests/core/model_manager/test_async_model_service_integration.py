# filename: test_async_model_service_integration_v1.py
# path: integration_tests/core/model_manager/test_async_model_service_integration.py
import pytest
import torch
import time

from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.core.model_manager.data.vram_monitor import PyTorchVramMonitor
from app.core.model_manager.utils.background_worker import BackgroundModelWorker
from app.core.model_manager.service.async_model_service import AsyncModelService
from app.core.model_manager.domain.interfaces import IModelLoader
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.core.model_manager import exceptions as exc


@pytest.fixture(scope="module", autouse=True)
def require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this system. Skipping real VRAM integration tests.")


class SlowDummyModelLoader(IModelLoader):
    """Simulates a model that takes a few seconds to load onto the GPU."""

    def load(self, manifest: ModelManifest) -> LoadedModel:
        time.sleep(1.0)  # Simulate disk I/O
        num_floats = int((manifest.required_vram_mb * 1024 * 1024) / 4)
        dummy_tensor = torch.zeros(num_floats, dtype=torch.float32, device="cuda")
        return LoadedModel(manifest=manifest, model_instance=dummy_tensor)

    def unload(self, model: LoadedModel) -> bool:
        del model.model_instance
        return True


@pytest.fixture
def async_service():
    monitor = PyTorchVramMonitor()
    orchestrator = ModelManagerOrchestrator(vram_monitor=monitor)
    worker = BackgroundModelWorker(orchestrator=orchestrator)
    service = AsyncModelService(worker=worker)
    yield service
    service.shutdown()


@pytest.mark.integration
def test_async_preload_overlaps_with_cpu_work(async_service):
    """
    Proves that the main thread can continue executing while the GPU model loads.
    """
    loader = SlowDummyModelLoader()
    manifest = ModelManifest(model_type=ModelType.VAD, model_path="fake", required_vram_mb=100.0)

    # 1. Trigger the background load
    start_time = time.time()
    future = async_service.trigger_preload(manifest, loader)

    # 2. Simulate doing CPU-bound work on the main thread (like FFmpeg extraction)
    # The loader takes 1.0 second. We will work for 1.2 seconds.
    time.sleep(1.2)
    cpu_work_time = time.time()

    # 3. Resolve the future. Because the model loaded while we were sleeping,
    # this should return almost instantly.
    loaded_model = async_service.resolve_preload(future)
    resolve_time = time.time()

    assert isinstance(loaded_model, LoadedModel)

    # Assert that the time spent resolving the future was near zero (< 0.1s)
    # proving it was ready and waiting for us.
    assert (resolve_time - cpu_work_time) < 0.1

    # Clean up
    async_service.trigger_unload(loaded_model, loader, flush=True)


@pytest.mark.integration
def test_async_preload_catches_out_of_vram(async_service):
    """
    Proves that hardware errors occurring on the background thread
    are correctly bubbled up to the main thread upon resolution.
    """
    loader = SlowDummyModelLoader()

    # Request an impossibly large model
    monitor = PyTorchVramMonitor()
    current_status = monitor.get_vram_status()
    impossible_vram = current_status.free_vram_mb + 1000.0

    manifest = ModelManifest(model_type=ModelType.TRANSCRIPTION, model_path="huge", required_vram_mb=impossible_vram)

    # 1. Trigger the load. This won't raise an error immediately.
    future = async_service.trigger_preload(manifest, loader)

    # 2. Wait for it to fail on the background thread and pull the error
    with pytest.raises(exc.OutOfVramError, match="Cannot load transcription"):
        async_service.resolve_preload(future)