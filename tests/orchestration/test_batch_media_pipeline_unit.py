# filename: test_batch_media_pipeline_unit_v1.py
import pytest
import uuid
from unittest.mock import Mock

from app.orchestration.service.pipelines.batch_media_pipeline import BatchMediaPipeline
from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import PhaseBatchResult, IngestJobResult
from app.orchestration.exceptions.workflow import WorkflowStateError
from app.features.storage.domain.entities import SourceType


class TestBatchMediaPipelineUnit:
    @pytest.fixture
    def mock_scan(self):
        return Mock()

    @pytest.fixture
    def mock_ingest(self):
        return Mock()

    @pytest.fixture
    def mock_extract(self):
        return Mock()

    @pytest.fixture
    def pipeline(self, mock_scan, mock_ingest, mock_extract):
        return BatchMediaPipeline(
            scan_phase=mock_scan,
            ingest_phase=mock_ingest,
            extract_audio_phase=mock_extract
        )

    def test_process_video_pipeline_perfect_run(self, pipeline, mock_scan, mock_ingest, mock_extract):
        """Verifies a VIDEO batch flows entirely through Scan -> Ingest -> Extract."""

        # 1. Setup Scan Result
        mock_scan.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.SCAN, status=PhaseStatus.COMPLETED,
            successful_count=1, items=["/vid.mp4"]
        )

        # 2. Setup Ingest Result (Returns a valid Source ID)
        valid_source_id = uuid.uuid4()
        mock_ingest.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.INGEST, status=PhaseStatus.COMPLETED,
            successful_count=1, items=[IngestJobResult(file_path="/vid.mp4", success=True, source_id=valid_source_id)]
        )

        # 3. Setup Extract Result
        mock_extract.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.AUDIO_EXTRACTION, status=PhaseStatus.COMPLETED, successful_count=1
        )

        # Execute
        results = pipeline.process_directory("/vault", SourceType.VIDEO)

        # Assertions
        assert PipelinePhase.SCAN in results
        assert PipelinePhase.INGEST in results
        assert PipelinePhase.AUDIO_EXTRACTION in results

        # Verify the pipeline specifically passed the successfully ingested source_id to Phase 3
        mock_extract.execute.assert_called_once_with([valid_source_id])

    def test_process_document_skips_audio_extraction(self, pipeline, mock_scan, mock_ingest, mock_extract):
        """Verifies that if SourceType is not VIDEO, Phase 3 is completely skipped."""
        mock_scan.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.SCAN, status=PhaseStatus.COMPLETED, successful_count=1, items=["/doc.txt"]
        )
        mock_ingest.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.INGEST, status=PhaseStatus.COMPLETED,
            successful_count=1, items=[IngestJobResult(file_path="/doc.txt", success=True, source_id=uuid.uuid4())]
        )

        results = pipeline.process_directory("/vault", SourceType.DOCUMENT)

        assert PipelinePhase.SCAN in results
        assert PipelinePhase.INGEST in results
        assert PipelinePhase.AUDIO_EXTRACTION not in results
        mock_extract.execute.assert_not_called()

    def test_pipeline_halts_on_scan_failure(self, pipeline, mock_scan, mock_ingest):
        """Verifies the state machine instantly raises an error if Phase 1 critically fails."""
        mock_scan.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.SCAN, status=PhaseStatus.FAILED, failed_count=1, errors=["Access Denied"]
        )

        with pytest.raises(WorkflowStateError, match="Scan phase failed"):
            pipeline.process_directory("/vault", SourceType.VIDEO)

        mock_ingest.execute.assert_not_called()

    def test_pipeline_filters_failed_ingests_before_extraction(self, pipeline, mock_scan, mock_ingest, mock_extract):
        """Verifies that if one video fails to ingest, it is NOT passed to Phase 3."""
        mock_scan.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.SCAN, status=PhaseStatus.COMPLETED, successful_count=2, items=["/v1.mp4", "/v2.mp4"]
        )

        valid_id = uuid.uuid4()
        # Mix of 1 success and 1 failure
        mock_ingest.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.INGEST, status=PhaseStatus.PARTIAL_SUCCESS,
            items=[
                IngestJobResult(file_path="/v1.mp4", success=True, source_id=valid_id),
                IngestJobResult(file_path="/v2.mp4", success=False, error_message="Disk Error")
            ]
        )

        # Mock extract
        mock_extract.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.AUDIO_EXTRACTION, status=PhaseStatus.COMPLETED
        )

        pipeline.process_directory("/vault", SourceType.VIDEO)

        # VERIFY: Phase 3 was called with ONLY the valid_id
        mock_extract.execute.assert_called_once_with([valid_id])