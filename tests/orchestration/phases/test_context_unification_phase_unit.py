# Canvas: test_context_unification_phase_unit_v2
# Path: tests/orchestration/phases/test_context_unification_phase_unit.py

import pytest
import uuid
from unittest.mock import Mock, ANY

from app.orchestration.service.phases.context_unification_phase import ContextUnificationPhase
from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import ContextualRetrievalJobResult
from app.features.contextual_retrieval.domain.entities import ContextualizedTranscript
from app.features.visual_context_synthesis.domain.entities import VisualTimeline
from app.features.context_unification.domain.entities import UnifiedTranscript


class TestContextUnificationPhaseUnit:
    @pytest.fixture
    def mock_job_manager(self): return Mock()

    @pytest.fixture
    def mock_unification_api(self): return Mock()

    @pytest.fixture
    def mock_contextual_repo(self): return Mock()

    @pytest.fixture
    def mock_visual_synthesis_repo(self): return Mock()

    @pytest.fixture
    def phase(
            self, mock_job_manager, mock_unification_api, mock_contextual_repo, mock_visual_synthesis_repo
    ):
        return ContextUnificationPhase(
            job_manager=mock_job_manager,
            unification_api=mock_unification_api,
            contextual_repo=mock_contextual_repo,
            visual_synthesis_repo=mock_visual_synthesis_repo
        )

    def test_execute_happy_path_with_visuals(self, phase, mock_job_manager, mock_contextual_repo,
                                             mock_visual_synthesis_repo, mock_unification_api):
        source_id = uuid.uuid4()
        upstream_results = [
            ContextualRetrievalJobResult(source_id=source_id, success=True, contextual_transcript_id=uuid.uuid4())
        ]

        # Setup Repositories
        mock_audio_transcript = Mock(spec=ContextualizedTranscript)
        mock_contextual_repo.get_by_source_id.return_value = mock_audio_transcript

        mock_visual_timeline = Mock(spec=VisualTimeline)
        mock_visual_synthesis_repo.get_by_source_id.return_value = mock_visual_timeline

        # Setup Job Manager Return
        mock_unified = Mock(spec=UnifiedTranscript)
        mock_unified.id = uuid.uuid4()
        mock_job_manager.run_job.return_value = mock_unified

        result = phase.execute(upstream_results)

        assert result.phase == PipelinePhase.CONTEXT_UNIFICATION
        assert result.status == PhaseStatus.COMPLETED
        assert result.successful_count == 1

        mock_job_manager.run_job.assert_called_once_with(
            ANY,
            mock_unification_api.execute,
            audio_transcript=mock_audio_transcript,
            visual_timeline=mock_visual_timeline
        )

    def test_execute_happy_path_audio_only(self, phase, mock_job_manager, mock_contextual_repo,
                                           mock_visual_synthesis_repo, mock_unification_api):
        """Ensures the phase safely processes audio-only files where visual timeline is None."""
        source_id = uuid.uuid4()
        upstream_results = [
            ContextualRetrievalJobResult(source_id=source_id, success=True, contextual_transcript_id=uuid.uuid4())
        ]

        mock_audio_transcript = Mock(spec=ContextualizedTranscript)
        mock_contextual_repo.get_by_source_id.return_value = mock_audio_transcript
        mock_visual_synthesis_repo.get_by_source_id.return_value = None

        mock_unified = Mock(spec=UnifiedTranscript)
        mock_unified.id = uuid.uuid4()  # <--- BUG FIX: Added the missing ID!
        mock_job_manager.run_job.return_value = mock_unified

        result = phase.execute(upstream_results)

        assert result.status == PhaseStatus.COMPLETED
        assert result.successful_count == 1

        mock_job_manager.run_job.assert_called_once_with(
            ANY,
            mock_unification_api.execute,
            audio_transcript=mock_audio_transcript,
            visual_timeline=None
        )

    def test_execute_skips_failed_upstream(self, phase, mock_job_manager):
        upstream_results = [
            ContextualRetrievalJobResult(source_id=uuid.uuid4(), success=False, error_message="Upstream fail")
        ]
        result = phase.execute(upstream_results)

        assert result.status == PhaseStatus.COMPLETED
        assert result.successful_count == 0
        mock_job_manager.run_job.assert_not_called()