# filename: test_orchestrator_integration_v2.py
# path: integration_tests/core/model_manager/test_orchestrator_integration.py
import pytest
import torch
import concurrent.futures

from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.core.model_manager.data.vram_monitor import PyTorchVramMonitor
from app.core.model_manager.domain.interfaces import IModelLoader
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.core.model_manager import exceptions as exc


@pytest.fixture(scope="module", autouse=True)
def require_cuda():
    """Skips all tests in this module if no NVIDIA GPU is detected."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this system. Skipping real VRAM integration tests.")


class DummyModelLoader(IModelLoader):
    def load(self, manifest: ModelManifest) -> LoadedModel:
        num_floats = int((manifest.required_vram_mb * 1024 * 1024) / 4)
        dummy_tensor = torch.zeros(num_floats, dtype=torch.float32, device="cuda")
        return LoadedModel(manifest=manifest, model_instance=dummy_tensor)

    def unload(self, model: LoadedModel) -> bool:
        del model.model_instance
        return True

    def predict(self, model: LoadedModel, inputs: dict) -> dict:
        return {"status": "dummy"}


@pytest.fixture
def orchestrator():
    monitor = PyTorchVramMonitor()
    return ModelManagerOrchestrator(vram_monitor=monitor)


@pytest.fixture
def dummy_loader():
    return DummyModelLoader()


@pytest.mark.integration
def test_real_vram_allocation_and_tracking(orchestrator, dummy_loader):
    baseline_status = orchestrator._monitor.get_vram_status()

    manifest = ModelManifest(
        model_type=ModelType.VAD,
        model_path="dummy_path",
        required_vram_mb=200.0
    )

    loaded_model = orchestrator.acquire_model(manifest, dummy_loader)
    loaded_status = orchestrator._monitor.get_vram_status()

    assert loaded_status.free_vram_mb < baseline_status.free_vram_mb
    assert loaded_model.id in orchestrator._active_models

    orchestrator.release_model(loaded_model, dummy_loader, flush=True)
    flushed_status = orchestrator._monitor.get_vram_status()

    assert flushed_status.free_vram_mb > loaded_status.free_vram_mb
    assert len(orchestrator._active_models) == 0


@pytest.mark.integration
def test_out_of_vram_rejection_real_math(orchestrator, dummy_loader):
    current_status = orchestrator._monitor.get_vram_status()
    impossible_vram_request = current_status.free_vram_mb + 100.0

    manifest = ModelManifest(
        model_type=ModelType.TRANSCRIPTION,
        model_path="too_big_model",
        required_vram_mb=impossible_vram_request
    )

    with pytest.raises(exc.OutOfVramError, match="Cannot load transcription") as error_info:
        orchestrator.acquire_model(manifest, dummy_loader)

    expected_vram = impossible_vram_request + orchestrator.VRAM_SAFETY_BUFFER_MB
    assert f"{expected_vram:.2f}" in str(error_info.value)


@pytest.mark.integration
def test_concurrent_vram_race_condition(orchestrator, dummy_loader):
    """
    Integration test to prove the Mutex lock prevents VRAM overallocation and
    corrupt dictionary state during concurrent background loading.
    """
    manifest_a = ModelManifest(model_type=ModelType.VAD, model_path="model_a", required_vram_mb=150.0)
    manifest_b = ModelManifest(model_type=ModelType.VISION, model_path="model_b", required_vram_mb=150.0)

    # Fire two loads in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(orchestrator.acquire_model, manifest_a, dummy_loader)
        future_b = executor.submit(orchestrator.acquire_model, manifest_b, dummy_loader)

        model_a = future_a.result()
        model_b = future_b.result()

    # Both should load successfully, but strictly sequentially
    assert len(orchestrator._active_models) == 2
    assert model_a.id in orchestrator._active_models
    assert model_b.id in orchestrator._active_models

    # Fire parallel unloads
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_ua = executor.submit(orchestrator.release_model, model_a, dummy_loader, flush=True)
        future_ub = executor.submit(orchestrator.release_model, model_b, dummy_loader, flush=True)

        future_ua.result()
        future_ub.result()

    # Verify safe cleanup
    assert len(orchestrator._active_models) == 0