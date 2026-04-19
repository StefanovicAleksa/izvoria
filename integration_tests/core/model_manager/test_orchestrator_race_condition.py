# Canvas: test_orchestrator_race_condition_v1
# Path: tests/core/model_manager/test_orchestrator_race_condition.py

import time
from unittest.mock import Mock

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.core.model_manager.utils.background_worker import BackgroundModelWorker
from app.core.model_manager.service.async_model_service import AsyncModelService


def test_background_preload_does_not_evict_active_model():
    """
    STRICT RACE CONDITION TEST.
    Proves that triggering a background preload for Phase N+1
    will spin-lock and wait politely, rather than brutally evicting Phase N's actively running model.
    """
    mock_monitor = Mock()
    mock_monitor.get_vram_status.return_value = Mock(free_vram_mb=24000.0)

    orchestrator = ModelManagerOrchestrator(mock_monitor)
    worker = BackgroundModelWorker(orchestrator)
    async_service = AsyncModelService(worker)

    try:
        # 1. Setup Phase 12 Model (Simulating Qwen-VL)
        manifest_12 = ModelManifest(model_type=ModelType.VISION, model_path="model_12", required_vram_mb=10)
        loader_12 = Mock()
        loader_12.load.return_value = LoadedModel(manifest=manifest_12, model_instance="ACTIVE_VISION_MODEL")

        # 2. Main Thread Acquires Phase 12 Model (Start of Phase 12 loop)
        model_12 = orchestrator.acquire_model(manifest_12, loader_12)
        assert model_12.model_instance == "ACTIVE_VISION_MODEL"

        # 3. Setup Phase 13 Model (Simulating Hallucination Gatekeeper)
        manifest_13 = ModelManifest(model_type=ModelType.VISUAL_HALLUCINATION_CORRECTION, model_path="model_13",
                                    required_vram_mb=10)
        loader_13 = Mock()
        loader_13.load.return_value = LoadedModel(manifest=manifest_13, model_instance="ACTIVE_SLM_MODEL")

        # 4. Trigger Phase 13 Preload (Background thread fires while Phase 12 is running)
        future = async_service.trigger_preload(manifest_13, loader_13)

        # 5. Let the background thread spin for a moment to attempt its hostile takeover
        time.sleep(0.5)

        # 6. CRITICAL ASSERTION: Phase 12 must STILL BE ACTIVE.
        # The Orchestrator must recognize it's a background request and spin-lock instead of evicting.
        assert orchestrator._active_models[model_12.id].model_instance == "ACTIVE_VISION_MODEL"
        assert not future.done()  # Future should be blocked waiting for Phase 12 to finish

        # 7. Main Thread finishes Phase 12 inference and triggers Unload
        async_service.trigger_unload(model_12, loader_12, flush=True)

        # 8. Background Thread should immediately wake up, acquire the lock, and load Phase 13
        model_13 = async_service.resolve_preload(future)

        assert model_13.model_instance == "ACTIVE_SLM_MODEL"

    finally:
        async_service.shutdown()