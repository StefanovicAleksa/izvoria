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


class _FakeGpu:
    """A VRAM monitor whose free memory moves as models load and unload, so eviction decisions are real."""

    def __init__(self, total_mb: float):
        self.total_mb = total_mb
        self.used: dict = {}
        self.flush_memory = MagicMock()
        self.reset_peak_stats = MagicMock()
        self.get_pytorch_memory_stats = MagicMock(return_value=(0.0, 0.0))

    def get_vram_status(self) -> VramStatus:
        used = sum(self.used.values())
        return VramStatus(total_vram_mb=self.total_mb, used_vram_mb=used, free_vram_mb=self.total_mb - used)

    def loader(self, footprint_mb: float) -> MagicMock:
        loader = MagicMock()

        def load(manifest):
            model = LoadedModel(manifest=manifest, model_instance=manifest.model_type.value)
            self.used[model.id] = footprint_mb
            return model

        loader.load.side_effect = load
        loader.unload.side_effect = lambda model: self.used.pop(model.id, None) is not None
        return loader


def _manifest(model_type: ModelType, estimate_mb: float) -> ModelManifest:
    # Paths that don't exist, so the estimate is required_vram_mb
    return ModelManifest(model_type=model_type, model_path=f"models/missing-{model_type.value}", required_vram_mb=estimate_mb)


WARM = [ModelType.RAG_EMBEDDING, ModelType.RAG_RERANKING, ModelType.QA_GENERATION]


class TestWarmResidency:
    """Keep-warm types stay resident after release and are evicted, least recently used first, only for room."""

    @pytest.fixture
    def gpu(self):
        return _FakeGpu(total_mb=12000.0)

    @pytest.fixture
    def orchestrator(self, gpu):
        orchestrator = ModelManagerOrchestrator(gpu, keep_warm_types=WARM)
        orchestrator.WAIT_POLL_INTERVAL_SEC = 0.01
        return orchestrator

    def test_a_released_warm_model_is_reused_without_loading(self, orchestrator, gpu):
        loader = gpu.loader(1000.0)
        manifest = _manifest(ModelType.RAG_EMBEDDING, 1000.0)
        first = orchestrator.acquire_model(manifest, loader, wait_for_clear=True)

        orchestrator.release_model(first, loader)
        again = orchestrator.acquire_model(manifest, loader, wait_for_clear=True)

        assert again is first
        loader.load.assert_called_once_with(manifest)
        loader.unload.assert_not_called()

    def test_a_type_outside_the_warm_set_still_unloads_on_release(self, orchestrator, gpu):
        loader = gpu.loader(1000.0)
        model = orchestrator.acquire_model(_manifest(ModelType.RETRIEVAL_SCRUBBER, 1000.0), loader, wait_for_clear=True)

        orchestrator.release_model(model, loader)

        loader.unload.assert_called_once_with(model)

    def test_a_chat_turn_keeps_embedder_reranker_and_answer_model_together(self, orchestrator, gpu):
        """The measured 12 GB dev-card set: 1090 + 1088 + 7788 MB, all three resident across turns."""
        roles = [(ModelType.RAG_EMBEDDING, 1090.0), (ModelType.RAG_RERANKING, 1088.0), (ModelType.QA_GENERATION, 7788.0)]
        loaders = {t: gpu.loader(mb) for t, mb in roles}
        manifests = {t: _manifest(t, mb) for t, mb in roles}

        for _turn in range(3):
            for model_type, _ in roles:
                model = orchestrator.acquire_model(manifests[model_type], loaders[model_type], wait_for_clear=True)
                orchestrator.release_model(model, loaders[model_type])

        for loader in loaders.values():
            assert loader.load.call_count == 1
            loader.unload.assert_not_called()

    def test_idle_models_are_evicted_least_recently_used_first_and_only_as_needed(self, orchestrator, gpu):
        a_loader, b_loader, c_loader = gpu.loader(4000.0), gpu.loader(4000.0), gpu.loader(5000.0)
        a = orchestrator.acquire_model(_manifest(ModelType.RAG_EMBEDDING, 4000.0), a_loader, wait_for_clear=True)
        b = orchestrator.acquire_model(_manifest(ModelType.RAG_RERANKING, 4000.0), b_loader, wait_for_clear=True)
        orchestrator.release_model(a, a_loader)
        orchestrator.release_model(b, b_loader)

        # 4000 free; 5000 + 500 buffer needs one eviction, and `a` went idle first
        orchestrator.acquire_model(_manifest(ModelType.QA_GENERATION, 5000.0), c_loader, wait_for_clear=True)

        a_loader.unload.assert_called_once_with(a)
        b_loader.unload.assert_not_called()

    def test_a_waiting_acquire_never_evicts_a_model_in_use(self, orchestrator, gpu):
        held_loader, next_loader = gpu.loader(8000.0), gpu.loader(5000.0)
        held = orchestrator.acquire_model(_manifest(ModelType.QA_GENERATION, 8000.0), held_loader, wait_for_clear=True)

        with ThreadPoolExecutor(max_workers=1) as pool:
            waiting = pool.submit(orchestrator.acquire_model, _manifest(ModelType.RAG_EMBEDDING, 5000.0),
                                  next_loader, wait_for_clear=True)
            time.sleep(0.3)
            assert not waiting.done()
            held_loader.unload.assert_not_called()

            # Released -> idle -> evictable
            orchestrator.release_model(held, held_loader)
            waiting.result(timeout=5)

        held_loader.unload.assert_called_once_with(held)
        next_loader.load.assert_called_once()

    def test_a_preload_loads_beside_a_model_in_use_when_it_fits(self, orchestrator, gpu):
        held_loader, next_loader = gpu.loader(1000.0), gpu.loader(1000.0)
        held = orchestrator.acquire_model(_manifest(ModelType.RAG_EMBEDDING, 1000.0), held_loader, wait_for_clear=True)

        orchestrator.acquire_model(_manifest(ModelType.RAG_RERANKING, 1000.0), next_loader, wait_for_clear=True)

        next_loader.load.assert_called_once()
        held_loader.unload.assert_not_called()
        assert orchestrator._find_resident(held.manifest) is held

    def test_a_batch_acquire_still_evicts_everything(self, orchestrator, gpu):
        warm_loader, batch_loader = gpu.loader(1000.0), gpu.loader(1000.0)
        warm = orchestrator.acquire_model(_manifest(ModelType.RAG_EMBEDDING, 1000.0), warm_loader, wait_for_clear=True)
        orchestrator.release_model(warm, warm_loader)

        orchestrator.acquire_model(_manifest(ModelType.VAD, 1000.0), batch_loader)

        warm_loader.unload.assert_called_once_with(warm)

    def test_release_all_unloads_idle_models_too(self, orchestrator, gpu):
        loader = gpu.loader(1000.0)
        model = orchestrator.acquire_model(_manifest(ModelType.RAG_EMBEDDING, 1000.0), loader, wait_for_clear=True)
        orchestrator.release_model(model, loader)

        orchestrator.release_all({})

        loader.unload.assert_called_once_with(model)
        assert gpu.used == {}

    def test_the_learned_footprint_replaces_the_estimate(self, orchestrator, gpu):
        """bge-m3 is 4.3 GB on disk and 1.1 GB loaded; once measured, the measurement is what is budgeted."""
        loader = gpu.loader(1100.0)
        manifest = _manifest(ModelType.RAG_EMBEDDING, 4300.0)
        orchestrator.acquire_model(manifest, loader, wait_for_clear=True)

        assert orchestrator._footprint_mb(manifest) == 1100.0

    def test_releasing_an_idle_model_again_is_harmless(self, orchestrator, gpu):
        loader = gpu.loader(1000.0)
        model = orchestrator.acquire_model(_manifest(ModelType.RAG_EMBEDDING, 1000.0), loader, wait_for_clear=True)

        orchestrator.release_model(model, loader)
        orchestrator.release_model(model, loader)

        loader.unload.assert_not_called()

    def test_an_abandoned_warm_preload_goes_idle_rather_than_unloading(self, orchestrator, gpu):
        loader = gpu.loader(1000.0)
        manifest = _manifest(ModelType.RAG_RERANKING, 1000.0)
        preloaded = orchestrator.acquire_model(manifest, loader, wait_for_clear=True)

        orchestrator.release_preload(preloaded, loader)

        loader.unload.assert_not_called()
        assert orchestrator.acquire_model(manifest, loader, wait_for_clear=True) is preloaded

    def test_backpack_is_learned_only_for_models_in_use(self, orchestrator, gpu):
        idle_loader, held_loader = gpu.loader(1000.0), gpu.loader(1000.0)
        idle = orchestrator.acquire_model(_manifest(ModelType.RAG_EMBEDDING, 1000.0), idle_loader, wait_for_clear=True)
        orchestrator.release_model(idle, idle_loader)
        held = orchestrator.acquire_model(_manifest(ModelType.RAG_RERANKING, 1000.0), held_loader, wait_for_clear=True)

        gpu.get_pytorch_memory_stats.return_value = (1000.0, 1800.0)
        orchestrator.record_telemetry()

        assert orchestrator.get_telemetry() == {held.manifest.model_path: 800.0}


class TestSingleSlotWithoutWarmTypes:
    """No keep-warm types is the pool as it always was."""

    def test_release_unloads_and_a_preload_waits_for_an_empty_gpu(self):
        gpu = _FakeGpu(total_mb=12000.0)
        orchestrator = ModelManagerOrchestrator(gpu)
        orchestrator.WAIT_POLL_INTERVAL_SEC = 0.01
        held_loader, next_loader = gpu.loader(1000.0), gpu.loader(1000.0)
        held = orchestrator.acquire_model(_manifest(ModelType.RAG_EMBEDDING, 1000.0), held_loader, wait_for_clear=True)

        with ThreadPoolExecutor(max_workers=1) as pool:
            waiting = pool.submit(orchestrator.acquire_model, _manifest(ModelType.RAG_RERANKING, 1000.0),
                                  next_loader, wait_for_clear=True)
            time.sleep(0.3)
            # It would fit, but a single-slot pool never loads beside a model in use
            assert not waiting.done()
            orchestrator.release_model(held, held_loader)
            waiting.result(timeout=5)

        held_loader.unload.assert_called_once_with(held)


class TestCheapestEviction:
    """Evict the fewest megabytes that make room; recency only breaks ties."""

    @pytest.fixture
    def gpu(self):
        return _FakeGpu(total_mb=12000.0)

    @pytest.fixture
    def orchestrator(self, gpu):
        return ModelManagerOrchestrator(gpu, keep_warm_types=WARM + [ModelType.REPORT_AUTHORING])

    def _idle(self, orchestrator, gpu, model_type, mb):
        loader = gpu.loader(mb)
        model = orchestrator.acquire_model(_manifest(model_type, mb), loader, wait_for_clear=True)
        orchestrator.release_model(model, loader)
        return model, loader

    def test_a_small_load_evicts_a_small_idle_model_not_an_older_large_one(self, orchestrator, gpu):
        """The report regression: the reranker evicted the 7.8 GB report model idle since scoping."""
        report, report_loader = self._idle(orchestrator, gpu, ModelType.REPORT_AUTHORING, 7800.0)
        embedder, embedder_loader = self._idle(orchestrator, gpu, ModelType.RAG_EMBEDDING, 1100.0)
        gpu.used["desktop"] = 2000.0  # 1100 free; the reranker needs 1100 + 500

        orchestrator.acquire_model(_manifest(ModelType.RAG_RERANKING, 1100.0), gpu.loader(1100.0),
                                   wait_for_clear=True)

        embedder_loader.unload.assert_called_once_with(embedder)
        report_loader.unload.assert_not_called()

    def test_a_large_load_evicts_the_one_large_model_and_keeps_the_small_ones(self, orchestrator, gpu):
        """The scrub turn: 4.4 GB of scrubber can only come from the answer model, so the embedder/reranker stay."""
        qa, qa_loader = self._idle(orchestrator, gpu, ModelType.QA_GENERATION, 7800.0)
        embedder, embedder_loader = self._idle(orchestrator, gpu, ModelType.RAG_EMBEDDING, 1100.0)
        reranker, reranker_loader = self._idle(orchestrator, gpu, ModelType.RAG_RERANKING, 1100.0)

        orchestrator.acquire_model(_manifest(ModelType.RETRIEVAL_SCRUBBER, 4400.0), gpu.loader(4400.0),
                                   wait_for_clear=True)

        qa_loader.unload.assert_called_once_with(qa)
        embedder_loader.unload.assert_not_called()
        reranker_loader.unload.assert_not_called()

    def test_when_no_single_model_is_enough_the_smallest_sufficient_set_goes(self, orchestrator, gpu):
        a, a_loader = self._idle(orchestrator, gpu, ModelType.RAG_EMBEDDING, 3000.0)
        b, b_loader = self._idle(orchestrator, gpu, ModelType.RAG_RERANKING, 3000.0)
        c, c_loader = self._idle(orchestrator, gpu, ModelType.QA_GENERATION, 5000.0)
        # 1000 free; 6000 + 500 needs 5500 more: {a, b} frees 6000, {a, c} or {b, c} 8000

        orchestrator.acquire_model(_manifest(ModelType.REPORT_AUTHORING, 6000.0), gpu.loader(6000.0),
                                   wait_for_clear=True)

        a_loader.unload.assert_called_once_with(a)
        b_loader.unload.assert_called_once_with(b)
        c_loader.unload.assert_not_called()


class TestTelemetryAttribution:
    @pytest.fixture
    def gpu(self):
        return _FakeGpu(total_mb=12000.0)

    def test_nothing_is_learned_while_two_models_are_in_use(self, gpu):
        orchestrator = ModelManagerOrchestrator(gpu, keep_warm_types=WARM)
        orchestrator.acquire_model(_manifest(ModelType.RAG_EMBEDDING, 1000.0), gpu.loader(1000.0), wait_for_clear=True)
        orchestrator.acquire_model(_manifest(ModelType.RAG_RERANKING, 1000.0), gpu.loader(1000.0), wait_for_clear=True)

        gpu.get_pytorch_memory_stats.return_value = (1000.0, 2084.0)
        orchestrator.record_telemetry()

        assert orchestrator.get_telemetry() == {}

    def test_the_counters_are_read_under_the_lock(self, gpu):
        """Loads and unloads hold the lock; a sample read outside it can land mid-eviction."""
        orchestrator = ModelManagerOrchestrator(gpu, keep_warm_types=WARM)
        orchestrator.acquire_model(_manifest(ModelType.RAG_EMBEDDING, 1000.0), gpu.loader(1000.0), wait_for_clear=True)
        owned = []
        gpu.get_pytorch_memory_stats.side_effect = lambda: owned.append(orchestrator._lock._is_owned()) or (0.0, 0.0)

        orchestrator.record_telemetry()

        assert owned == [True]


PINNED = [ModelType.RAG_EMBEDDING, ModelType.RAG_RERANKING]
EXTERNAL = [ModelType.QA_GENERATION, ModelType.RETRIEVAL_SCRUBBER]


class TestPinnedAndExternalResidency:
    """The hosted API's models: pinned ones are loaded once and never freed; external ones live in llama-server."""

    @pytest.fixture
    def gpu(self):
        return _FakeGpu(total_mb=24000.0)

    @pytest.fixture
    def orchestrator(self, gpu):
        orchestrator = ModelManagerOrchestrator(gpu, pinned_types=PINNED, external_types=EXTERNAL,
                                                strict_resident=True)
        orchestrator.WAIT_POLL_INTERVAL_SEC = 0.01
        return orchestrator

    def test_a_type_cannot_be_both_pinned_and_external(self, gpu):
        with pytest.raises(ValueError, match="rag_embedding"):
            ModelManagerOrchestrator(gpu, pinned_types=[ModelType.RAG_EMBEDDING],
                                     external_types=[ModelType.RAG_EMBEDDING])

    def test_a_pinned_model_survives_release_release_all_and_a_batch_acquire(self, gpu):
        orchestrator = ModelManagerOrchestrator(gpu, pinned_types=PINNED)
        loader = gpu.loader(1090.0)
        embedder = orchestrator.warm_up(_manifest(ModelType.RAG_EMBEDDING, 1090.0), loader)

        orchestrator.release_model(embedder, loader)
        orchestrator.release_preload(embedder, loader)
        orchestrator.release_all({})
        batch_loader = gpu.loader(3500.0)
        orchestrator.acquire_model(_manifest(ModelType.TRANSCRIPTION, 3500.0), batch_loader)

        loader.unload.assert_not_called()
        assert orchestrator.acquire_model(_manifest(ModelType.RAG_EMBEDDING, 1090.0), loader) is embedder
        loader.load.assert_called_once()

    def test_an_external_model_is_connected_once_without_budgeting_or_flushing(self, orchestrator, gpu):
        gpu.used["somebody-else"] = 23900.0  # the card is full; llama-server's VRAM is not ours to budget
        loader = MagicMock()
        loader.load.side_effect = lambda m: LoadedModel(manifest=m, model_instance="client")
        manifest = _manifest(ModelType.QA_GENERATION, 10000.0)

        first = orchestrator.acquire_model(manifest, loader, wait_for_clear=True)
        second = orchestrator.acquire_model(manifest, loader, wait_for_clear=True)
        orchestrator.release_model(first, loader)
        orchestrator.release_all({})

        assert second is first
        loader.load.assert_called_once_with(manifest)
        loader.unload.assert_not_called()
        gpu.flush_memory.assert_not_called()

    def test_strict_mode_refuses_a_type_that_is_neither_pinned_nor_external(self, orchestrator, gpu):
        loader = gpu.loader(5000.0)
        with pytest.raises(exc.ResidencyError, match="vision"):
            orchestrator.acquire_model(_manifest(ModelType.VISION, 5000.0), loader, wait_for_clear=True)
        loader.load.assert_not_called()

    def test_warm_up_refuses_a_type_that_would_leak(self, orchestrator, gpu):
        with pytest.raises(ValueError, match="vad"):
            orchestrator.warm_up(_manifest(ModelType.VAD, 100.0), gpu.loader(100.0))

    def test_a_waiting_acquire_does_not_wait_for_pinned_models(self, gpu):
        """Single-slot pool plus pinned models: a live acquire must not park behind a model nobody releases."""
        orchestrator = ModelManagerOrchestrator(gpu, pinned_types=PINNED)
        orchestrator.WAIT_POLL_INTERVAL_SEC = 0.01
        orchestrator.warm_up(_manifest(ModelType.RAG_EMBEDDING, 1090.0), gpu.loader(1090.0))

        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(orchestrator.acquire_model, _manifest(ModelType.QA_REASONER, 7800.0),
                                 gpu.loader(7800.0), True)
            assert future.result(timeout=2).manifest.model_type is ModelType.QA_REASONER

    def test_concurrent_acquires_share_one_instance_and_never_unload(self, orchestrator, gpu):
        loader = gpu.loader(1090.0)
        manifest = _manifest(ModelType.RAG_EMBEDDING, 1090.0)
        orchestrator.warm_up(manifest, loader)

        def use():
            model = orchestrator.acquire_model(manifest, loader, wait_for_clear=True)
            orchestrator.release_model(model, loader)
            return model

        with ThreadPoolExecutor(8) as pool:
            models = list(pool.map(lambda _: use(), range(64)))

        assert len({m.id for m in models}) == 1
        loader.unload.assert_not_called()

    def test_shutdown_all_frees_pinned_models(self, orchestrator, gpu):
        loader = gpu.loader(1090.0)
        embedder = orchestrator.warm_up(_manifest(ModelType.RAG_EMBEDDING, 1090.0), loader)

        orchestrator.shutdown_all({})

        loader.unload.assert_called_once_with(embedder)
        assert orchestrator.resident_types() == []

    def test_release_all_leaves_the_cuda_cache_alone_when_only_permanent_models_are_resident(self, orchestrator, gpu):
        orchestrator.warm_up(_manifest(ModelType.RAG_EMBEDDING, 1090.0), gpu.loader(1090.0))
        gpu.flush_memory.reset_mock()

        orchestrator.release_all({})

        gpu.flush_memory.assert_not_called()


    def test_resident_types_lists_what_is_loaded(self, orchestrator, gpu):
        orchestrator.warm_up(_manifest(ModelType.RAG_EMBEDDING, 1090.0), gpu.loader(1090.0))
        client = MagicMock()
        client.load.side_effect = lambda m: LoadedModel(manifest=m, model_instance="client")
        orchestrator.acquire_model(_manifest(ModelType.QA_GENERATION, 1.0), client)

        assert sorted(t.value for t in orchestrator.resident_types()) == ["qa_generation", "rag_embedding"]

    def test_a_failed_external_connect_registers_nothing_and_is_retried(self, orchestrator):
        loader = MagicMock()
        loader.load.side_effect = [RuntimeError("llama-server down"),
                                   LoadedModel(manifest=_manifest(ModelType.QA_GENERATION, 1.0), model_instance="c")]
        manifest = _manifest(ModelType.QA_GENERATION, 1.0)

        with pytest.raises(exc.ModelManagerError):
            orchestrator.acquire_model(manifest, loader, wait_for_clear=True)
        assert orchestrator.resident_types() == []

        assert orchestrator.acquire_model(manifest, loader, wait_for_clear=True).model_instance == "c"

    def test_warming_an_external_type_connects_it(self, orchestrator):
        loader = MagicMock()
        loader.load.side_effect = lambda m: LoadedModel(manifest=m, model_instance="client")
        manifest = _manifest(ModelType.RETRIEVAL_SCRUBBER, 1.0)

        warmed = orchestrator.warm_up(manifest, loader)

        assert orchestrator.acquire_model(manifest, loader, wait_for_clear=True) is warmed
        loader.load.assert_called_once_with(manifest)


class TestWarmUpAdmission:
    def test_a_pinned_model_is_admitted_by_its_declared_requirement_not_its_disk_size(self, tmp_path):
        """bge-m3: 4.3 GB on disk, 1.1 GB loaded. Beside llama-server, the disk estimate refused it."""
        weights = tmp_path / "bge-m3"
        weights.mkdir()
        (weights / "big.bin").write_bytes(b"\0" * (3 * 1024 ** 2))  # 3 MB on disk
        gpu = _FakeGpu(total_mb=2.0 + 0.5 + ModelManagerOrchestrator.VRAM_SAFETY_BUFFER_MB)
        orchestrator = ModelManagerOrchestrator(gpu, pinned_types=[ModelType.RAG_EMBEDDING])
        manifest = ModelManifest(model_type=ModelType.RAG_EMBEDDING, model_path=str(weights), required_vram_mb=2.0)

        model = orchestrator.warm_up(manifest, gpu.loader(1.0))

        assert model.manifest is manifest

    def test_a_pinned_model_that_cannot_fit_fails_at_startup(self):
        gpu = _FakeGpu(total_mb=1000.0)
        orchestrator = ModelManagerOrchestrator(gpu, pinned_types=[ModelType.RAG_EMBEDDING])
        with pytest.raises(exc.OutOfVramError, match="pin rag_embedding"):
            orchestrator.warm_up(_manifest(ModelType.RAG_EMBEDDING, 2500.0), gpu.loader(1090.0))
