# Canvas: test_frame_extraction_phase_unit_v1
# Path: tests/orchestration/service/phases/test_frame_extraction_phase_unit.py

import pytest
import uuid
from unittest.mock import Mock

from app.orchestration.service.phases.frame_extraction_phase import FrameExtractionPhase
from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import VisualCueDetectionJobResult, FrameExtractionJobResult
from app.features.contextual_retrieval.domain.entities import ContextualizedTranscript
from app.features.storage.domain.entities import Source, File, SourceType


class TestFrameExtractionPhaseUnit:
    @pytest.fixture
    def mock_job_manager(self): return Mock()

    @pytest.fixture
    def mock_extraction_api(self): return Mock()

    @pytest.fixture
    def mock_contextual_repo(self): return Mock()

    @pytest.fixture
    def mock_storage(self): return Mock()

    @pytest.fixture
    def phase(
            self, mock_job_manager, mock_extraction_api, mock_contextual_repo, mock_storage
    ):
        return FrameExtractionPhase(
            job_manager=mock_job_manager,
            extraction_api=mock_extraction_api,
            contextual_repo=mock_contextual_repo,
            storage_api=mock_storage
        )

    def test_execute_happy_path(self, phase, mock_job_manager, mock_contextual_repo, mock_storage):
        source_id = uuid.uuid4()
        upstream_results = [
            VisualCueDetectionJobResult(source_id=source_id, success=True, evaluated_transcript_id=uuid.uuid4())
        ]

        # Setup Storage
        mock_file = File(id=uuid.uuid4(), file_path="/vault/vid.mp4", file_name="v", file_size_bytes=100)
        mock_source = Source(id=source_id, name="v", source_type=SourceType.VIDEO, content_hash="123", file_id=mock_file.id, file=mock_file)
        mock_storage.get_source.return_value = mock_source

        # Setup Transcript
        mock_contextual_repo.get_by_source_id.return_value = Mock(spec=ContextualizedTranscript)

        # Setup JobManager Return Value
        mock_job_manager.run_job.return_value = {"extracted_chunks_count": 5}

        result = phase.execute(upstream_results)

        assert result.phase == PipelinePhase.FRAME_EXTRACTION
        assert result.status == PhaseStatus.COMPLETED
        assert result.successful_count == 1
        assert result.failed_count == 0
        assert isinstance(result.items[0], FrameExtractionJobResult)
        assert result.items[0].success is True
        assert result.items[0].extracted_chunks_count == 5

    def test_execute_skips_failed_upstream(self, phase):
        upstream_results = [
            VisualCueDetectionJobResult(source_id=uuid.uuid4(), success=False, error_message="Upstream fail")
        ]

        result = phase.execute(upstream_results)

        assert result.status == PhaseStatus.COMPLETED
        assert result.successful_count == 0
        assert result.failed_count == 0