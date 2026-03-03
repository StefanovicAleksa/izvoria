# Canvas: test_batch_media_pipeline_unit_v5
# Path: tests/orchestration/service/pipelines/test_batch_media_pipeline_unit.py

import pytest
import uuid
from unittest.mock import Mock

from app.orchestration.service.pipelines.batch_media_pipeline import BatchMediaPipeline
from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import (
    PhaseBatchResult,
    IngestJobResult,
    AudioExtractionJobResult,
    VadJobResult,
    TranscribeJobResult,
    DiarizeJobResult,
    IntelligenceBridgeJobResult
)
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
    def mock_transcribe(self): return Mock()

    @pytest.fixture
    def mock_diarize(self): return Mock()

    @pytest.fixture
    def mock_bridge(self): return Mock()

    @pytest.fixture
    def pipeline(self, mock_scan, mock_ingest, mock_extract, mock_vad, mock_transcribe, mock_diarize, mock_bridge):
        return BatchMediaPipeline(
            scan_phase=mock_scan,
            ingest_phase=mock_ingest,
            extract_audio_phase=mock_extract,
            vad_phase=mock_vad,
            transcribe_phase=mock_transcribe,
            diarize_phase=mock_diarize,
            intelligence_bridge_phase=mock_bridge
        )

    def test_process_video_pipeline_perfect_run(self, pipeline, mock_scan, mock_ingest, mock_extract, mock_vad,
                                                mock_transcribe, mock_diarize, mock_bridge):
        """Verifies a VIDEO batch flows entirely through Scan -> Ingest -> Extract -> VAD -> Transcribe -> Diarize -> Intelligence Bridge."""
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

        vad_items = [VadJobResult(source_id=valid_source_id, success=True, vad_map_id=uuid.uuid4())]
        mock_vad.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.VAD, status=PhaseStatus.COMPLETED, successful_count=1, items=vad_items
        )

        transcribe_items = [TranscribeJobResult(source_id=valid_source_id, success=True, transcript_id=uuid.uuid4())]
        mock_transcribe.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.TRANSCRIBE, status=PhaseStatus.COMPLETED, successful_count=1, items=transcribe_items
        )

        diarize_items = [DiarizeJobResult(source_id=valid_source_id, success=True, diarization_map_id=uuid.uuid4())]
        mock_diarize.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.DIARIZE, status=PhaseStatus.COMPLETED, successful_count=1, items=diarize_items
        )

        bridge_items = [IntelligenceBridgeJobResult(source_id=valid_source_id, success=True, scene_map_id=uuid.uuid4())]
        mock_bridge.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.INTELLIGENCE_BRIDGE, status=PhaseStatus.COMPLETED, successful_count=1,
            items=bridge_items
        )

        results = pipeline.process_directory("/vault", SourceType.VIDEO)

        assert PipelinePhase.SCAN in results
        assert PipelinePhase.INGEST in results
        assert PipelinePhase.AUDIO_EXTRACTION in results
        assert PipelinePhase.VAD in results
        assert PipelinePhase.TRANSCRIBE in results
        assert PipelinePhase.DIARIZE in results
        assert PipelinePhase.INTELLIGENCE_BRIDGE in results

        # Verify async preloads were called before execution
        mock_vad.preload_model.assert_called_once()
        mock_transcribe.preload_model.assert_called_once()
        mock_diarize.preload_model.assert_called_once()
        mock_bridge.preload_model.assert_called_once()

        # Verify Phases were executed with the output of the previous Phase
        mock_vad.execute.assert_called_once_with(extracted_items)
        mock_transcribe.execute.assert_called_once_with(vad_items)
        mock_diarize.execute.assert_called_once_with(transcribe_items)
        mock_bridge.execute.assert_called_once_with(diarize_items)

    def test_process_document_skips_audio_extraction(self, pipeline, mock_scan, mock_ingest, mock_extract, mock_vad,
                                                     mock_transcribe, mock_diarize, mock_bridge):
        """Verifies that if SourceType is not VIDEO, audio/vision/NLP phases are completely skipped."""
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
        assert PipelinePhase.TRANSCRIBE not in results
        assert PipelinePhase.DIARIZE not in results
        assert PipelinePhase.INTELLIGENCE_BRIDGE not in results

        mock_extract.execute.assert_not_called()
        mock_vad.preload_model.assert_not_called()
        mock_transcribe.execute.assert_not_called()
        mock_diarize.execute.assert_not_called()
        mock_bridge.execute.assert_not_called()