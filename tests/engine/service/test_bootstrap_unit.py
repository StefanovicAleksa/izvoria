# Canvas: test_bootstrap_unit_v3
# Path: tests/engine/service/test_bootstrap_unit.py

import pytest
from unittest.mock import Mock, call, patch

from app.engine.exceptions.lifecycle import EngineBusyError
from app.engine.service.bootstrap import IzvoriaBootstrapper
from app.engine.service.izvoria_engine import IzvoriaEngine


class TestIzvoriaBootstrapperUnit:

    @patch('app.core.database.service.manager.db_manager')
    @patch('app.features.scanner.service.scanner.FileScanner')
    @patch('app.core.model_manager.data.vram_monitor.PyTorchVramMonitor')
    def test_bootstrapper_builds_engine_successfully(self, mock_vram, mock_scanner, mock_db):
        """
        Validates that the massive dependency injection container can assemble
        all repositories, adapters, APIs, and phases without raising an exception.
        """
        try:
            engine = IzvoriaBootstrapper.build()
        except Exception as e:
            pytest.fail(f"Bootstrapper failed to assemble dependencies: {e}")

        try:
            assert isinstance(engine, IzvoriaEngine)
            assert engine._media_pipeline is not None
            assert engine._audio_pipeline is not None
            assert engine._live_pipeline is not None
            assert engine._live_verified_pipeline is not None
            assert engine._live_qa_pipeline is not None
            assert engine._live_reasoning_pipeline is not None
            assert engine._session_memory_api is not None
        finally:
            # The container starts a real BackgroundModelWorker whose telemetry thread polls
            # torch.cuda every 0.5s; left running, it leaks into later tests that patch torch.cuda.
            engine._async_service.shutdown()

class TestSwitchLanguage:
    @pytest.fixture
    def settings(self):
        with patch('app.engine.service.bootstrap.settings') as settings:
            settings.APPLIANCE_LANGUAGE = "en"
            yield settings

    def test_stops_switches_rebuilds_starts_and_remembers_in_that_order(self, settings):
        calls = Mock()
        old = calls.old
        settings.use_language.side_effect = lambda code: calls.use_language(code)
        settings.remember_language.side_effect = lambda code: calls.remember_language(code)
        listener = Mock()
        with patch.object(IzvoriaBootstrapper, 'build', side_effect=lambda progress_listener: calls.build(progress_listener)):
            rebuilt = IzvoriaBootstrapper.switch_language(old, "sr", progress_listener=listener)

        assert calls.mock_calls == [
            call.old.require_idle(), call.old.stop(), call.use_language("sr"), call.build(listener),
            call.build().start(), call.remember_language("sr"),
        ]
        assert rebuilt is calls.build.return_value

    def test_a_busy_engine_is_left_alone(self, settings):
        old = Mock()
        old.require_idle.side_effect = EngineBusyError("ingesting")

        with patch.object(IzvoriaBootstrapper, 'build') as build, pytest.raises(EngineBusyError):
            IzvoriaBootstrapper.switch_language(old, "sr")

        old.stop.assert_not_called()
        settings.use_language.assert_not_called()
        build.assert_not_called()

    def test_a_failed_rebuild_goes_back_to_the_previous_language_and_remembers_nothing(self, settings):
        with patch.object(IzvoriaBootstrapper, 'build', side_effect=RuntimeError("no GPU")), \
                pytest.raises(RuntimeError):
            IzvoriaBootstrapper.switch_language(Mock(), "sr")

        assert settings.use_language.mock_calls == [call("sr"), call("en")]
        settings.remember_language.assert_not_called()

    def test_without_an_engine_it_just_builds_one(self, settings):
        with patch.object(IzvoriaBootstrapper, 'build') as build:
            rebuilt = IzvoriaBootstrapper.switch_language(None, "sl")

        settings.use_language.assert_called_once_with("sl")
        build.return_value.start.assert_called_once_with()
        assert rebuilt is build.return_value


class TestModelKit:
    """The hosted API's shared model service: one kit, one engine per firm."""

    def test_the_appliance_kit_has_no_pinned_external_or_strict_residency(self):
        kit = IzvoriaBootstrapper.build_model_kit()
        try:
            orchestrator = kit.orchestrator
            assert orchestrator._pinned_types == frozenset()
            assert orchestrator._external_types == frozenset()
            assert orchestrator._strict_resident is False
        finally:
            kit.async_service.shutdown()

    def test_hosted_settings_reach_the_orchestrator(self, monkeypatch):
        from app.core.config import settings
        from app.core.model_manager.data.headroom_vram_monitor import HeadroomVramMonitor
        from app.core.model_manager.domain.enums import ModelType
        monkeypatch.setattr(settings, "MODEL_PINNED_TYPES", ["rag_embedding", "rag_reranking"])
        monkeypatch.setattr(settings, "MODEL_EXTERNAL_TYPES", ["qa_generation"])
        monkeypatch.setattr(settings, "ENGINE_CONCURRENT_LIVE", True)
        monkeypatch.setattr(settings, "VRAM_HEADROOM_MB", 1200.0)

        kit = IzvoriaBootstrapper.build_model_kit()
        try:
            orchestrator = kit.orchestrator
            assert orchestrator._pinned_types == {ModelType.RAG_EMBEDDING, ModelType.RAG_RERANKING}
            assert orchestrator._external_types == {ModelType.QA_GENERATION}
            assert orchestrator._strict_resident is True
            assert isinstance(orchestrator._monitor, HeadroomVramMonitor)
        finally:
            kit.async_service.shutdown()

    @patch('app.core.database.service.manager.db_manager')
    def test_firm_engines_share_the_kit_but_not_the_database(self, _):
        kit = IzvoriaBootstrapper.build_model_kit()
        first_db, second_db = Mock(), Mock()
        try:
            first = IzvoriaBootstrapper.build(database=first_db, kit=kit)
            second = IzvoriaBootstrapper.build(database=second_db, kit=kit)

            assert first._async_service is kit.async_service is second._async_service
            assert first._db is first_db and second._db is second_db
            assert first._owns_model_service is False
            assert first._retrieval_api._search_repo._db is first_db
            assert second._retrieval_api._search_repo._db is second_db
        finally:
            kit.async_service.shutdown()

    def test_warm_up_loads_only_pinned_and_external_types(self, monkeypatch):
        from app.core.config import settings
        from app.core.model_manager.domain.enums import ModelType
        from app.engine.service.bootstrap import ModelKit
        monkeypatch.setattr(settings, "MODEL_PINNED_TYPES", ["rag_embedding"])
        monkeypatch.setattr(settings, "MODEL_EXTERNAL_TYPES", ["qa_generation", "retrieval_scrubber"])
        orchestrator = Mock()
        kit = ModelKit(orchestrator=orchestrator, worker=Mock(), async_service=Mock(), prompt_adapter=Mock(),
                       reranker_adapter=Mock(), scrubber_adapter=Mock(), qa_adapter=Mock(), reasoner_adapter=Mock(),
                       report_adapter=Mock(), hf_embedder=Mock())

        warmed = kit.warm_up()

        assert warmed == [ModelType.RAG_EMBEDDING, ModelType.QA_GENERATION, ModelType.RETRIEVAL_SCRUBBER]
        loaders = [c.args[1] for c in orchestrator.warm_up.call_args_list]
        assert loaders == [kit.prompt_adapter, kit.qa_adapter, kit.scrubber_adapter]
        assert orchestrator.warm_up.call_args_list[0].args[0].model_path == settings.RAG_EMBEDDING_MODEL_PATH

    @patch('app.core.database.service.manager.db_manager')
    def test_a_firm_engine_keeps_its_files_in_the_firms_own_vault_and_reports(self, _, tmp_path):
        kit = IzvoriaBootstrapper.build_model_kit()
        try:
            engine = IzvoriaBootstrapper.build(database=Mock(), kit=kit, vault_dir=tmp_path / "vault",
                                            reports_dir=tmp_path / "reports")

            assert engine._storage_api.file_manager.base_path == (tmp_path / "vault").resolve()
            assert engine._report_api._file_store._base_dir == tmp_path / "reports"
        finally:
            kit.async_service.shutdown()

    @patch('app.engine.service.bootstrap.db_manager')
    def test_the_appliance_engine_owns_its_own_kit_and_database(self, appliance_db):
        engine = IzvoriaBootstrapper.build()
        try:
            assert engine._owns_model_service is True
            assert engine._db is appliance_db
            assert engine._report_api._file_store._base_dir is None
        finally:
            engine._async_service.shutdown()

    def test_warm_up_skips_a_reranker_without_weights(self, monkeypatch, tmp_path):
        from app.core.config import settings
        from app.engine.service.bootstrap import ModelKit
        monkeypatch.setattr(settings, "MODEL_PINNED_TYPES", ["rag_reranking"])
        monkeypatch.setattr(settings, "MODEL_EXTERNAL_TYPES", [])
        monkeypatch.setattr(settings, "RAG_RERANKER_MODEL_PATH", str(tmp_path / "missing"))
        orchestrator = Mock()
        kit = ModelKit(orchestrator=orchestrator, worker=Mock(), async_service=Mock(), prompt_adapter=Mock(),
                       reranker_adapter=Mock(), scrubber_adapter=Mock(), qa_adapter=Mock(), reasoner_adapter=Mock(),
                       report_adapter=Mock(), hf_embedder=Mock())

        assert kit.warm_up() == []
        orchestrator.warm_up.assert_not_called()

    def test_shutdown_stops_the_preload_worker_then_frees_every_model(self):
        from app.engine.service.bootstrap import ModelKit
        order = Mock()
        kit = ModelKit(orchestrator=order.orchestrator, worker=Mock(), async_service=order.async_service,
                       prompt_adapter=Mock(), reranker_adapter=Mock(), scrubber_adapter=Mock(), qa_adapter=Mock(),
                       reasoner_adapter=Mock(), report_adapter=Mock(), hf_embedder=Mock())

        kit.shutdown()

        assert order.mock_calls == [call.async_service.shutdown(), call.orchestrator.shutdown_all({})]
