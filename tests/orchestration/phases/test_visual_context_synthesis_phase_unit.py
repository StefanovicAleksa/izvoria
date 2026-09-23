# Canvas: test_visual_context_synthesis_phase_unit_v1
# Path: tests/orchestration/phases/test_visual_context_synthesis_phase_unit.py

import pytest
import uuid
from unittest.mock import Mock

from app.orchestration.service.phases.visual_context_synthesis_phase import VisualContextSynthesisPhase
from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import VisualHallucinationCorrectionJobResult
from app.core.model_manager.domain.entities import LoadedModel
from app.features.contextual_retrieval.domain.entities import ContextualizedTranscript


class TestVisualContextSynthesisPhaseUnit:
    @pytest.fixture
    def mock_job_manager(self): return Mock()

    @pytest.fixture
    def mock_synthesis_api(self): return Mock()

    @pytest.fixture
    def mock_contextual_repo(self): return Mock()

    @pytest.fixture
    def mock_async_service(self): return Mock()

    @pytest.fixture
    def mock_loader(self): return Mock()

    @pytest.fixture
    def phase(
            self, mock_job_manager, mock_synthesis_api, mock_contextual_repo,
            mock_async_service, mock_loader
    ):
        return VisualContextSynthesisPhase(
            job_manager=mock_job_manager,
            synthesis_api=mock_synthesis_api,
            contextual_repo=mock_contextual_repo,
            async_model_service=mock_async_service,
            loader=mock_loader
        )

    def test_preload_model_triggers_async_service(self, phase, mock_async_service):
        """Verifies that preloading correctly registers a future in the phase state."""
        mock_future = Mock()
        mock_async_service.trigger_preload.return_value = mock_future

        phase.preload_model()

        assert phase._preload_future == mock_future
        mock_async_service.trigger_preload.assert_called_once()

    def test_execute_success_flow(self, phase, mock_job_manager, mock_async_service, mock_contextual_repo):
        """Verifies the happy path processes correctly and clears VRAM."""
        source_id = uuid.uuid4()
        prev_results = [VisualHallucinationCorrectionJobResult(source_id=source_id, success=True)]

        # Mock Preload and Resolution
        mock_future = Mock()
        mock_async_service.trigger_preload.return_value = mock_future

        mock_model = Mock(spec=LoadedModel)
        mock_async_service.resolve_preload.return_value = mock_model

        # Mock Transcript Retrieval
        mock_contextual_repo.get_by_source_id.return_value = Mock(spec=ContextualizedTranscript)

        # Mock Job Manager execution
        mock_job_manager.run_job.return_value = Mock(id=uuid.uuid4())

        # Trigger Preload
        phase.preload_model()

        # Execute Phase
        result = phase.execute(prev_results)

        # Assertions
        assert result.phase == PipelinePhase.VISUAL_CONTEXT_SYNTHESIS
        assert result.status == PhaseStatus.COMPLETED
        assert result.successful_count == 1

        # Verify GPU lifecycle is handled
        mock_async_service.resolve_preload.assert_called_once_with(mock_future)
        mock_async_service.trigger_unload.assert_called_once_with(
            loaded_model=mock_model,
            loader=phase._loader,
            flush=True
        )

    def test_execute_skips_failed_upstream_items(self, phase, mock_job_manager):
        """Verifies that the phase ignores transcripts that failed in Phase 13."""
        source_id = uuid.uuid4()
        prev_results = [VisualHallucinationCorrectionJobResult(source_id=source_id, success=False,
                                                               error_message="Gatekeeper Crash")]

        result = phase.execute(prev_results)

        assert result.successful_count == 0
        mock_job_manager.run_job.assert_not_called()