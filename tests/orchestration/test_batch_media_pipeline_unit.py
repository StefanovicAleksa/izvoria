# filename: test_batch_media_pipeline_unit_v2.py
# path: tests/orchestration/service/pipelines/test_batch_media_pipeline_unit.py
import pytest
import uuid
from unittest.mock import Mock

from app.orchestration.service.pipelines.batch_media_pipeline import BatchMediaPipeline
from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import PhaseBatchResult, IngestJobResult, AudioExtractionJobResult
from app.features.storage.domain.entities import SourceType


class TestBatchMediaPipelineUnit:
    @pytest.fixture
    def mock_scan(self): return Mock()

    @pytest.fixture
    def mock_ingest(self): return Mock()

    @pytest.fixture
    def mock_extract(self): return Mock()

    @pytest.fixture
    def mock_vad(self): return Mock()

    @pytest.fixture
    def pipeline(self, mock_scan, mock_ingest, mock_extract, mock_vad):
        return BatchMediaPipeline(
            scan_phase=mock_scan,
            ingest_phase=mock_ingest,
            extract_audio_phase=mock_extract,
            vad_phase=mock_vad
        )

    def test_process_video_pipeline_perfect_run(self, pipeline, mock_scan, mock_ingest, mock_extract, mock_vad):
        """Verifies a VIDEO batch flows entirely through Scan -> Ingest -> Extract -> VAD."""

        mock_scan.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.SCAN, status=PhaseStatus.COMPLETED,
            successful_count=1, items=["/vid.mp4"]
        )

        valid_source_id = uuid.uuid4()
        mock_ingest.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.INGEST, status=PhaseStatus.COMPLETED,
            successful_count=1, items=[IngestJobResult(file_path="/vid.mp4", success=True, source_id=valid_source_id)]
        )

        extracted_items = [
            AudioExtractionJobResult(source_id=valid_source_id, success=True, extracted_file_id=uuid.uuid4())]
        mock_extract.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.AUDIO_EXTRACTION, status=PhaseStatus.COMPLETED, successful_count=1,
            items=extracted_items
        )

        mock_vad.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.VAD, status=PhaseStatus.COMPLETED, successful_count=1
        )

        results = pipeline.process_directory("/vault", SourceType.VIDEO)

        assert PipelinePhase.SCAN in results
        assert PipelinePhase.INGEST in results
        assert PipelinePhase.AUDIO_EXTRACTION in results
        assert PipelinePhase.VAD in results

        # Verify async preload was called before extraction
        mock_vad.preload_model.assert_called_once()

        # Verify VAD was executed with the output of extraction
        mock_vad.execute.assert_called_once_with(extracted_items)

    def test_process_document_skips_audio_extraction(self, pipeline, mock_scan, mock_ingest, mock_extract, mock_vad):
        """Verifies that if SourceType is not VIDEO, Phase 3 and 4 are completely skipped."""
        mock_scan.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.SCAN, status=PhaseStatus.COMPLETED, successful_count=1, items=["/doc.txt"]
        )
        mock_ingest.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.INGEST, status=PhaseStatus.COMPLETED,
            successful_count=1, items=[IngestJobResult(file_path="/doc.txt", success=True, source_id=uuid.uuid4())]
        )

        results = pipeline.process_directory("/vault", SourceType.DOCUMENT)

        assert PipelinePhase.AUDIO_EXTRACTION not in results
        assert PipelinePhase.VAD not in results
        mock_extract.execute.assert_not_called()
        mock_vad.preload_model.assert_not_called()