# Canvas: test_visual_hallucination_correction_phase_unit_v2
# Path: tests/orchestration/phases/test_visual_hallucination_correction_phase_unit.py

import pytest
import uuid
from unittest.mock import Mock

from app.orchestration.service.phases.visual_hallucination_correction_phase import VisualHallucinationCorrectionPhase
from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import VisualAnalysisJobResult
from app.core.model_manager.domain.entities import LoadedModel


class TestVisualHallucinationCorrectionPhaseUnit:
    @pytest.fixture
    def mock_job_manager(self): return Mock()

    @pytest.fixture
    def mock_correction_api(self): return Mock()

    @pytest.fixture
    def mock_async_service(self): return Mock()

    @pytest.fixture
    def mock_loader(self): return Mock()

    @pytest.fixture
    def phase(
            self, mock_job_manager, mock_correction_api,
            mock_async_service, mock_loader
    ):
        return VisualHallucinationCorrectionPhase(
            job_manager=mock_job_manager,
            correction_api=mock_correction_api,
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

    def test_execute_success_flow(self, phase, mock_job_manager, mock_async_service):
        """FIXED: Now triggers a preload before execution to satisfy the GPU lifecycle assertion."""
        # 1. Setup Source Data
        source_id = uuid.uuid4()
        prev_results = [VisualAnalysisJobResult(source_id=source_id, success=True, analyzed_chunks_count=1)]

        # 2. Mock Preload and Resolution
        mock_future = Mock()
        mock_async_service.trigger_preload.return_value = mock_future

        mock_model = Mock(spec=LoadedModel)
        mock_async_service.resolve_preload.return_value = mock_model

        # 3. Mock Job Manager execution
        mock_job_manager.run_job.return_value = Mock(id=uuid.uuid4())

        # 4. Trigger Preload (sets self._preload_future)
        phase.preload_model()

        # 5. Execute Phase
        result = phase.execute(prev_results)

        # Assertions
        assert result.phase == PipelinePhase.VISUAL_HALLUCINATION_CORRECTION
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
        """Verifies that the phase ignores transcripts that failed in Phase 12."""
        # Setup
        source_id = uuid.uuid4()
        prev_results = [VisualAnalysisJobResult(source_id=source_id, success=False, error_message="VLM Crash")]

        # Execute
        result = phase.execute(prev_results)

        # Assertions
        assert result.successful_count == 0
        mock_job_manager.run_job.assert_not_called()

    def test_execute_works_without_preload(self, phase, mock_job_manager, mock_async_service):
        """Verifies that the phase can still execute even if background preloading was skipped."""
        source_id = uuid.uuid4()
        prev_results = [VisualAnalysisJobResult(source_id=source_id, success=True, analyzed_chunks_count=1)]
        mock_job_manager.run_job.return_value = Mock(id=uuid.uuid4())

        # Execute without calling phase.preload_model()
        result = phase.execute(prev_results)

        assert result.status == PhaseStatus.COMPLETED
        mock_async_service.resolve_preload.assert_not_called()
        mock_async_service.trigger_unload.assert_not_called()