# Canvas: test_clip_extraction_phase_unit_v1
# Path: tests/orchestration/phases/test_clip_extraction_phase_unit.py

import pytest
import uuid
from unittest.mock import Mock, ANY

from app.orchestration.service.phases.clip_extraction_phase import ClipExtractionPhase
from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import ContextUnificationJobResult, ClipExtractionJobResult
from app.features.context_unification.domain.entities import UnifiedTranscript


class TestClipExtractionPhaseUnit:
    @pytest.fixture
    def mock_job_manager(self): return Mock()

    @pytest.fixture
    def mock_extractor_api(self): return Mock()

    @pytest.fixture
    def mock_unified_repo(self): return Mock()

    @pytest.fixture
    def phase(
            self, mock_job_manager, mock_extractor_api, mock_unified_repo
    ):
        return ClipExtractionPhase(
            job_manager=mock_job_manager,
            extractor_api=mock_extractor_api,
            unified_repo=mock_unified_repo
        )

    def test_execute_happy_path(self, phase, mock_job_manager, mock_unified_repo, mock_extractor_api):
        source_id = uuid.uuid4()
        upstream_results = [
            ContextUnificationJobResult(source_id=source_id, success=True, unified_transcript_id=uuid.uuid4())
        ]

        # Setup Repositories
        mock_transcript = Mock(spec=UnifiedTranscript)
        mock_unified_repo.get_by_source_id.return_value = mock_transcript

        # Setup Job Manager Return (a list of extracted VideoClips)
        mock_job_manager.run_job.return_value = [Mock(), Mock()]

        result = phase.execute(upstream_results)

        assert result.phase == PipelinePhase.CLIP_EXTRACTION
        assert result.status == PhaseStatus.COMPLETED
        assert result.successful_count == 1
        assert result.failed_count == 0

        mock_job_manager.run_job.assert_called_once_with(
            ANY,
            mock_extractor_api.extract_clips_from_transcript,
            transcript=mock_transcript
        )

        assert isinstance(result.items[0], ClipExtractionJobResult)
        assert result.items[0].extracted_clips_count == 2

    def test_execute_skips_failed_upstream(self, phase, mock_job_manager):
        upstream_results = [
            ContextUnificationJobResult(source_id=uuid.uuid4(), success=False, error_message="Upstream fail")
        ]
        result = phase.execute(upstream_results)

        assert result.status == PhaseStatus.COMPLETED
        assert result.successful_count == 0
        mock_job_manager.run_job.assert_not_called()