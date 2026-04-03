# Canvas: test_contextual_retrieval_phase_unit_v3
# Path: tests/orchestration/phases/test_contextual_retrieval_phase_unit.py

import pytest
import uuid
import concurrent.futures
from unittest.mock import Mock, ANY

from app.orchestration.service.phases.contextual_retrieval_phase import ContextualRetrievalPhase
from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import PhoneticCorrectionJobResult
from app.core.model_manager.domain.entities import LoadedModel
from app.features.phonetic_correction.domain.entities import PhoneticCorrectedTranscript
from app.features.contextual_retrieval.domain.entities import ContextualizedTranscript


class TestContextualRetrievalPhaseUnit:
    @pytest.fixture
    def mock_job_manager(self): return Mock()

    @pytest.fixture
    def mock_contextualizer_api(self): return Mock()

    @pytest.fixture
    def mock_phonetic_repo(self): return Mock()

    @pytest.fixture
    def mock_async_service(self): return Mock()

    @pytest.fixture
    def mock_loader(self): return Mock()

    @pytest.fixture
    def phase(
            self, mock_job_manager, mock_contextualizer_api, mock_phonetic_repo,
            mock_async_service, mock_loader
    ):
        # FIXED: Removed `manifest` and `adapter` overrides, replaced with strict `loader` injection.
        return ContextualRetrievalPhase(
            job_manager=mock_job_manager,
            contextualizer_api=mock_contextualizer_api,
            phonetic_repo=mock_phonetic_repo,
            async_model_service=mock_async_service,
            loader=mock_loader
        )

    def test_preload_model_triggers_async_service(self, phase, mock_async_service, mock_loader):
        phase.preload_model()
        mock_async_service.trigger_preload.assert_called_once_with(manifest=ANY, loader=mock_loader)

    def test_execute_happy_path(self, phase, mock_async_service, mock_job_manager, mock_phonetic_repo, mock_loader):
        source_id = uuid.uuid4()
        upstream_results = [
            PhoneticCorrectionJobResult(source_id=source_id, success=True, phonetic_transcript_id=uuid.uuid4())
        ]

        mock_future = Mock(spec=concurrent.futures.Future)
        phase._preload_future = mock_future
        mock_loaded_model = Mock(spec=LoadedModel)
        mock_async_service.resolve_preload.return_value = mock_loaded_model

        mock_phonetic_repo.get_by_source_id.return_value = Mock(spec=PhoneticCorrectedTranscript)

        mock_contextual_transcript = Mock(spec=ContextualizedTranscript)
        mock_contextual_transcript.id = uuid.uuid4()
        mock_job_manager.run_job.return_value = mock_contextual_transcript

        result = phase.execute(upstream_results)

        assert result.phase == PipelinePhase.CONTEXTUAL_RETRIEVAL
        assert result.status == PhaseStatus.COMPLETED
        assert result.successful_count == 1

        mock_async_service.trigger_unload.assert_called_once_with(loaded_model=mock_loaded_model, loader=mock_loader,
                                                                  flush=True)

    def test_execute_skips_failed_upstream(self, phase):
        phase._preload_future = None
        upstream_results = [
            PhoneticCorrectionJobResult(source_id=uuid.uuid4(), success=False, error_message="Upstream fail")
        ]
        result = phase.execute(upstream_results)
        assert result.status == PhaseStatus.COMPLETED
        assert result.successful_count == 0

    def test_execute_fails_if_preload_fails(self, phase, mock_async_service):
        mock_future = Mock(spec=concurrent.futures.Future)
        phase._preload_future = mock_future
        from app.core.model_manager.exceptions.base import ModelManagerError
        mock_async_service.resolve_preload.side_effect = ModelManagerError("Out of VRAM")

        result = phase.execute([])
        assert result.status == PhaseStatus.FAILED