# Canvas: test_orchestrator_integration_v3
# Path: integration_tests/core/model_manager/test_orchestrator_integration.py

import os
import time
from typing import Optional

import pytest
import torch

from app.core.config import settings
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.core.model_manager.data.vram_monitor import PyTorchVramMonitor
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.core.model_manager.utils.background_worker import BackgroundModelWorker
from app.features.vad.data.nemo_adapter import NeMoVadAdapter


class TestModelManagerIntegration:
    """
    STRICT NO-MOCKS INTEGRATION TEST.
    Requires: NVIDIA GPU and the local MarbleNet VAD model on disk.
    Verifies dynamic size discovery and background telemetry tracking.
    """

    @pytest.fixture(scope="module", autouse=True)
    def ensure_hardware(self):
        if not torch.cuda.is_available():
            pytest.skip("Integration test requires an active NVIDIA GPU.")

    @pytest.fixture
    def vad_manifest(self) -> ModelManifest:
        manifest = ModelManifest(
            model_type=ModelType.VAD,
            model_path=settings.VAD_MODEL_PATH,
            required_vram_mb=settings.VAD_VRAM_REQUIREMENT_MB
        )
        if not os.path.exists(manifest.model_path):
            pytest.skip(f"VAD model missing at {manifest.model_path}. Cannot run integration test.")
        return manifest

    def test_dynamic_size_discovery_and_telemetry_tracking(self, vad_manifest):
        # 1. Boot up the live system
        monitor = PyTorchVramMonitor()
        monitor.flush_memory()  # Ensure clean slate

        initial_status = monitor.get_vram_status()

        orchestrator = ModelManagerOrchestrator(monitor)
        worker = BackgroundModelWorker(orchestrator)
        adapter = NeMoVadAdapter()

        loaded_model: Optional[LoadedModel] = None

        try:
            # 2. Acquire the Model (Tests disk size auto-discovery)
            loaded_model = orchestrator.acquire_model(vad_manifest, adapter)

            assert loaded_model.id in orchestrator._active_models

            # 3. Simulate an Inference Memory Spike (The "Backpack")
            dummy_inference_tensor = torch.ones((1024, 1024, 100), device="cuda", dtype=torch.float32)

            # Ensure PyTorch executes the allocation
            torch.cuda.synchronize()

            # 4. Free the simulated inference spike, returning model to resting state
            del dummy_inference_tensor

            # 5. Wait for the Background Telemetry Daemon to poll the resting state
            time.sleep(1.0)

            # 6. Verify the Orchestrator dynamically learned the Backpack
            learned_backpack = orchestrator._historical_backpacks.get(vad_manifest.model_path, 0.0)

            print(
                f"\n[TEST INFO] Discovered Weight Size: {orchestrator._get_actual_size_mb(vad_manifest.model_path, 0):.2f} MB")
            print(f"[TEST INFO] Learned Inference Backpack: {learned_backpack:.2f} MB")

            assert learned_backpack > 350.0, "Telemetry daemon failed to record the memory spike!"

        finally:
            # 7. Tear down and ensure strict VRAM cleanup
            worker.shutdown(wait=True)

            if loaded_model:
                orchestrator.release_all(loaders_map={loaded_model.id: adapter})

            final_status = monitor.get_vram_status()

            # Account for ~50MB of PyTorch CUDA context overhead that never unloads
            assert final_status.used_vram_mb <= initial_status.used_vram_mb + 100.0, \
                "VRAM leak detected! Memory did not return to baseline."