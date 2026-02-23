# File: integration_tests/core/model_manager/test_orchestrator_integration.py
import pytest
import torch

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
    """
    A fake adapter that allocates raw tensors on the GPU to simulate
    the VRAM footprint of loading a real AI model.
    """

    def load(self, manifest: ModelManifest) -> LoadedModel:
        # 1 Float32 = 4 Bytes.
        # To allocate exactly 100MB: (100 * 1024 * 1024) / 4 = 26,214,400 floats
        # We allocate a bit more to ensure it registers clearly above background noise.
        num_floats = int((manifest.required_vram_mb * 1024 * 1024) / 4)

        # Move the junk tensor to the GPU immediately
        dummy_tensor = torch.zeros(num_floats, dtype=torch.float32, device="cuda")

        return LoadedModel(manifest=manifest, model_instance=dummy_tensor)

    def unload(self, model: LoadedModel) -> bool:
        # Delete the python reference to the tensor so the garbage collector can find it
        del model.model_instance
        return True

    def predict(self, model: LoadedModel, inputs: dict) -> dict:
        return {"status": "dummy"}


@pytest.fixture
def orchestrator():
    """Wires up the real Service with the real PyTorch Data adapter."""
    monitor = PyTorchVramMonitor()
    return ModelManagerOrchestrator(vram_monitor=monitor)


@pytest.fixture
def dummy_loader():
    return DummyModelLoader()


@pytest.mark.integration
def test_real_vram_allocation_and_tracking(orchestrator, dummy_loader):
    """
    Verifies that the Orchestrator successfully reads the hardware changes
    when a model is moved into GPU memory.
    """
    # 1. Take a baseline reading
    baseline_status = orchestrator._monitor.get_vram_status()

    # 2. Command the Orchestrator to load a fake 200MB model
    manifest = ModelManifest(
        model_type=ModelType.VAD,
        model_path="dummy_path",
        required_vram_mb=200.0
    )

    loaded_model = orchestrator.acquire_model(manifest, dummy_loader)

    # 3. Verify VRAM usage increased at the hardware level
    loaded_status = orchestrator._monitor.get_vram_status()

    # It won't be EXACTLY 200MB because PyTorch's caching allocator grabs chunks,
    # but the free VRAM should definitively be lower than the baseline.
    assert loaded_status.free_vram_mb < baseline_status.free_vram_mb
    assert loaded_model.id in orchestrator._active_models

    # 4. Release the model
    orchestrator.release_model(loaded_model, dummy_loader, flush=True)

    # 5. Verify the hardware was aggressively flushed back to near-baseline levels
    flushed_status = orchestrator._monitor.get_vram_status()
    assert flushed_status.free_vram_mb > loaded_status.free_vram_mb
    assert len(orchestrator._active_models) == 0


@pytest.mark.integration
def test_out_of_vram_rejection_real_math(orchestrator, dummy_loader):
    """
    Verifies the bouncer correctly calculates available memory from the real GPU
    and rejects models that would cause a CUDA Out-of-Memory crash.
    """
    current_status = orchestrator._monitor.get_vram_status()

    # Create a manifest that requests exactly 100MB MORE than the GPU physically has left
    impossible_vram_request = current_status.free_vram_mb + 100.0

    manifest = ModelManifest(
        model_type=ModelType.TRANSCRIPTION,
        model_path="too_big_model",
        required_vram_mb=impossible_vram_request
    )

    # FIX: Updated the match string to reflect the Orchestrator's preemptive warning
    with pytest.raises(exc.OutOfVramError, match="Cannot load transcription") as error_info:
        orchestrator.acquire_model(manifest, dummy_loader)

    # Ensure the error message contains our safety buffer logic, formatted exactly as the Orchestrator does it.
    expected_vram = impossible_vram_request + orchestrator.VRAM_SAFETY_BUFFER_MB
    assert f"{expected_vram:.2f}" in str(error_info.value)