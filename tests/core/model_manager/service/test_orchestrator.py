import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from unittest.mock import MagicMock, patch

from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel, VramStatus
from app.core.model_manager.domain.enums import ModelType
from app.core.model_manager import exceptions as exc


class TestModelManagerOrchestrator:
    @pytest.fixture
    def mock_monitor(self):
        monitor = MagicMock()
        # Assume 12GB total, 10GB free by default
        monitor.get_vram_status.return_value = VramStatus(
            total_vram_mb=12000.0, used_vram_mb=2000.0, free_vram_mb=10000.0
        )
        monitor.get_pytorch_memory_stats.return_value = (0.0, 0.0)
        return monitor

    @pytest.fixture
    def mock_loader(self):
        return MagicMock()

    @pytest.fixture
    def sample_manifest(self):
        return ModelManifest(
            model_type=ModelType.VAD,
            model_path="models/dummy_model.bin",
            required_vram_mb=1000.0
        )

    def test_acquire_model_success_and_idempotent(self, mock_monitor, mock_loader, sample_manifest):
        orchestrator = ModelManagerOrchestrator(mock_monitor)

        dummy_loaded = LoadedModel(manifest=sample_manifest, model_instance="fake_tensor")
        mock_loader.load.return_value = dummy_loaded

        # 1. First Load
        result1 = orchestrator.acquire_model(sample_manifest, mock_loader)

        # FIXED: Verify pre-emptive flush happened
        mock_monitor.flush_memory.assert_called_once()
        assert result1 == dummy_loaded
        mock_loader.load.assert_called_once()
        mock_monitor.reset_peak_stats.assert_called_once()

        # 2. Second Load (Idempotency)
        result2 = orchestrator.acquire_model(sample_manifest, mock_loader)
        assert result2 == dummy_loaded
        assert mock_loader.load.call_count == 1  # Should NOT call load again

    def test_acquire_model_blocks_oom(self, mock_monitor, mock_loader, sample_manifest):
        orchestrator = ModelManagerOrchestrator(mock_monitor)

        # Mock VRAM to only have 100MB free
        mock_monitor.get_vram_status.return_value = VramStatus(
            total_vram_mb=12000.0, used_vram_mb=11900.0, free_vram_mb=100.0
        )

        with pytest.raises(exc.OutOfVramError) as e_info:
            orchestrator.acquire_model(sample_manifest, mock_loader)

        # FIXED: Verify pre-emptive flush happened even before OOM check
        mock_monitor.flush_memory.assert_called_once()
        assert "Insufficient VRAM for vad" in str(e_info.value)
        mock_loader.load.assert_not_called()

    @patch("app.core.model_manager.service.orchestrator.Path")
    def test_get_actual_size_mb_dynamic_discovery(self, mock_path, mock_monitor):
        """Proves the orchestrator checks the OS for real file sizes."""
        orchestrator = ModelManagerOrchestrator(mock_monitor)

        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.is_file.return_value = True
        mock_path_instance.stat.return_value.st_size = 2048 * (1024 ** 2)
        mock_path.return_value = mock_path_instance

        size = orchestrator._get_actual_size_mb("models/dummy.gguf", 1000.0)
        assert size == 2048.0

    def test_record_telemetry_learns_backpack_spike(self, mock_monitor, mock_loader, sample_manifest):
        orchestrator = ModelManagerOrchestrator(mock_monitor)

        dummy_loaded = LoadedModel(manifest=sample_manifest, model_instance="fake_tensor")
        mock_loader.load.return_value = dummy_loaded
        orchestrator.acquire_model(sample_manifest, mock_loader)

        # Simulate a telemetry spike (Current: 1000MB, Peak: 2500MB -> Backpack: 1500MB)
        mock_monitor.get_pytorch_memory_stats.return_value = (1000.0, 2500.0)
        orchestrator.record_telemetry()

        assert orchestrator._historical_backpacks[sample_manifest.model_path] == 1500.0

    def test_release_all_emergency_flush(self, mock_monitor, mock_loader, sample_manifest):
        orchestrator = ModelManagerOrchestrator(mock_monitor)

        dummy_loaded = LoadedModel(manifest=sample_manifest, model_instance="fake")
        mock_loader.load.return_value = dummy_loaded
        orchestrator.acquire_model(sample_manifest, mock_loader)

        # Reset mock to clear pre-emptive calls from acquire_model
        mock_monitor.flush_memory.reset_mock()

        loaders_map = {dummy_loaded.id: mock_loader}
        orchestrator.release_all(loaders_map)

        mock_loader.unload.assert_called_once_with(dummy_loaded)
        mock_monitor.flush_memory.assert_called_once()

    def test_same_path_different_model_type_is_not_reused(self, mock_monitor, sample_manifest):
        """The same weights loaded by a different feature must be swapped, never handed over."""
        orchestrator = ModelManagerOrchestrator(mock_monitor)
        first_loader, second_loader = MagicMock(), MagicMock()

        first_loaded = LoadedModel(manifest=sample_manifest, model_instance="ctx_4096")
        first_loader.load.return_value = first_loaded
        orchestrator.acquire_model(sample_manifest, first_loader)

        other_manifest = ModelManifest(
            model_type=ModelType.CONTEXTUAL_RETRIEVAL,
            model_path=sample_manifest.model_path,
            required_vram_mb=1000.0
        )
        second_loaded = LoadedModel(manifest=other_manifest, model_instance="ctx_32768")
        second_loader.load.return_value = second_loaded

        result = orchestrator.acquire_model(other_manifest, second_loader)

        assert result == second_loaded
        second_loader.load.assert_called_once_with(other_manifest)
        first_loader.unload.assert_called_once_with(first_loaded)

    def test_wait_for_clear_waits_as_long_as_the_resident_model_is_in_use(self, mock_monitor, mock_loader,
                                                                          sample_manifest):
        """A preload never evicts a model in use and never gives up on it: it loads once that model is released."""
        orchestrator = ModelManagerOrchestrator(mock_monitor)
        orchestrator.WAIT_POLL_INTERVAL_SEC = 0.01
        resident = LoadedModel(manifest=sample_manifest, model_instance="in use")
        mock_loader.load.return_value = resident
        orchestrator.acquire_model(sample_manifest, mock_loader)

        other_manifest = ModelManifest(
            model_type=ModelType.TRANSCRIPTION,
            model_path="models/other.bin",
            required_vram_mb=1000.0
        )
        other_loader = MagicMock()
        other_loaded = LoadedModel(manifest=other_manifest, model_instance="next")
        other_loader.load.return_value = other_loaded

        with ThreadPoolExecutor(max_workers=1) as pool:
            preload = pool.submit(orchestrator.acquire_model, other_manifest, other_loader, wait_for_clear=True)
            time.sleep(0.3)
            assert not preload.done()
            other_loader.load.assert_not_called()
            mock_loader.unload.assert_not_called()

            orchestrator.release_model(resident, mock_loader)

            assert preload.result(timeout=5) == other_loaded
        other_loader.load.assert_called_once_with(other_manifest)

    def test_wait_for_clear_returns_resident_of_same_manifest_immediately(
            self, mock_monitor, mock_loader, sample_manifest
    ):
        orchestrator = ModelManagerOrchestrator(mock_monitor)
        loaded = LoadedModel(manifest=sample_manifest, model_instance="resident")
        mock_loader.load.return_value = loaded
        orchestrator.acquire_model(sample_manifest, mock_loader)

        result = orchestrator.acquire_model(sample_manifest, mock_loader, wait_for_clear=True)

        assert result == loaded
        assert mock_loader.load.call_count == 1


class TestReleasePreload:
    """
    `release_preload` is the fire-and-forget release used by `discard_preload`'s done-callback.

    Every other release is made by the code holding the model, which knows it has finished. This one
    runs on a background thread on behalf of a caller that walked away, and by then the model may have
    been handed to somebody else -- freeing it there is a use-after-free that aborts the process inside
    llama.cpp's `free_model` rather than raising.
    """

    @pytest.fixture
    def mock_monitor(self):
        monitor = MagicMock()
        monitor.get_vram_status.return_value = VramStatus(
            total_vram_mb=12000.0, used_vram_mb=2000.0, free_vram_mb=10000.0
        )
        monitor.get_pytorch_memory_stats.return_value = (0.0, 0.0)
        return monitor

    @pytest.fixture
    def manifest(self):
        return ModelManifest(
            model_type=ModelType.VAD, model_path="models/dummy_model.bin", required_vram_mb=1000.0
        )

    @pytest.fixture
    def loader(self, manifest):
        loader = MagicMock()
        loader.load.return_value = LoadedModel(manifest=manifest, model_instance="fake_tensor")
        return loader

    def test_an_abandoned_preload_nobody_took_is_released(self, mock_monitor, loader, manifest):
        orchestrator = ModelManagerOrchestrator(mock_monitor)
        preloaded = orchestrator.acquire_model(manifest, loader, wait_for_clear=True)

        orchestrator.release_preload(preloaded, loader)

        loader.unload.assert_called_once_with(preloaded)
        assert orchestrator.acquire_model(manifest, loader) is not None
        assert loader.load.call_count == 2, "the model should have to be loaded again"

    def test_a_preload_somebody_else_acquired_is_left_resident(self, mock_monitor, loader, manifest):
        """The crash: the callback must not free the instance another caller is mid-generation on."""
        orchestrator = ModelManagerOrchestrator(mock_monitor)
        preloaded = orchestrator.acquire_model(manifest, loader, wait_for_clear=True)

        # A phase resolves the preload and takes the model -- served straight from the idempotency check.
        in_use = orchestrator.acquire_model(manifest, loader)
        assert in_use is preloaded

        orchestrator.release_preload(preloaded, loader)

        loader.unload.assert_not_called()
        assert orchestrator.acquire_model(manifest, loader) is preloaded
        assert loader.load.call_count == 1, "the model must still be resident"

    def test_the_holder_can_still_release_it_afterwards(self, mock_monitor, loader, manifest):
        orchestrator = ModelManagerOrchestrator(mock_monitor)
        preloaded = orchestrator.acquire_model(manifest, loader, wait_for_clear=True)
        in_use = orchestrator.acquire_model(manifest, loader)

        orchestrator.release_preload(preloaded, loader)
        orchestrator.release_model(in_use, loader)

        loader.unload.assert_called_once_with(in_use)
        orchestrator.acquire_model(manifest, loader)
        assert loader.load.call_count == 2, "a discarded-then-released model must not stay resident"

    def test_releasing_a_preload_twice_is_harmless(self, mock_monitor, loader, manifest):
        orchestrator = ModelManagerOrchestrator(mock_monitor)
        preloaded = orchestrator.acquire_model(manifest, loader, wait_for_clear=True)

        orchestrator.release_preload(preloaded, loader)
        orchestrator.release_preload(preloaded, loader)

        assert loader.unload.call_count == 1

    def test_a_model_that_was_already_evicted_is_ignored(self, mock_monitor, loader, manifest):
        orchestrator = ModelManagerOrchestrator(mock_monitor)
        preloaded = orchestrator.acquire_model(manifest, loader, wait_for_clear=True)
        orchestrator.release_all({})
        loader.unload.reset_mock()

        orchestrator.release_preload(preloaded, loader)

        loader.unload.assert_not_called()

    def test_the_hold_pattern_still_frees_on_one_release(self, mock_monitor, loader, manifest):
        """
        The report path's shape: a Phase resolves one preload, the feature API acquires the same
        manifest once per job (`release_after=False`, so it never releases), and the Phase's `finally`
        releases once. Counting hand-outs must not turn that into a leak.
        """
        orchestrator = ModelManagerOrchestrator(mock_monitor)
        held = orchestrator.acquire_model(manifest, loader, wait_for_clear=True)
        for _ in range(5):
            assert orchestrator.acquire_model(manifest, loader) is held

        orchestrator.release_model(held, loader)

        loader.unload.assert_called_once_with(held)
        orchestrator.acquire_model(manifest, loader)
        assert loader.load.call_count == 2, "one release must free it however many times it was handed out"
