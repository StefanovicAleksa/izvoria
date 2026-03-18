# Canvas: test_batch_media_pipeline_unit_v9
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
    DomainAdaptationJobResult,
    TranscribeJobResult,
    DiarizeJobResult,
    SentenceReconstructionJobResult,
    PhoneticCorrectionJobResult
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
    def mock_domain_adaptation(self): return Mock()

    @pytest.fixture
    def mock_transcribe(self): return Mock()

    @pytest.fixture
    def mock_diarize(self): return Mock()

    @pytest.fixture
    def mock_sentence(self): return Mock()

    @pytest.fixture
    def mock_phonetic(self): return Mock()

    @pytest.fixture
    def pipeline(self, mock_scan, mock_ingest, mock_extract, mock_vad, mock_domain_adaptation, mock_transcribe,
                 mock_diarize, mock_sentence, mock_phonetic):
        return BatchMediaPipeline(
            scan_phase=mock_scan,
            ingest_phase=mock_ingest,
            extract_audio_phase=mock_extract,
            vad_phase=mock_vad,
            domain_adaptation_phase=mock_domain_adaptation,
            transcribe_phase=mock_transcribe,
            diarize_phase=mock_diarize,
            sentence_reconstruction_phase=mock_sentence,
            phonetic_correction_phase=mock_phonetic
        )

    def test_pipeline_happy_path(self, pipeline, mock_scan, mock_ingest, mock_extract, mock_vad, mock_domain_adaptation,
                                 mock_transcribe, mock_diarize, mock_sentence, mock_phonetic):
        """Tests that a fully successful run flows perfectly through all 8 phases."""
        target_dir = "/dummy/path"
        source_id = uuid.uuid4()

        # Mock the sequential phase outputs
        mock_scan.execute.return_value = PhaseBatchResult(phase=PipelinePhase.SCAN, status=PhaseStatus.COMPLETED,
                                                          successful_count=1, items=["/dummy/path/vid.mp4"])
        mock_ingest.execute.return_value = PhaseBatchResult(phase=PipelinePhase.INGEST, status=PhaseStatus.COMPLETED,
                                                            items=[IngestJobResult(source_id=source_id, success=True,
                                                                                   file_path="/dummy/path/vid.mp4")])
        mock_extract.execute.return_value = PhaseBatchResult(phase=PipelinePhase.AUDIO_EXTRACTION,
                                                             status=PhaseStatus.COMPLETED, items=[
                AudioExtractionJobResult(source_id=source_id, success=True)])
        mock_vad.execute.return_value = PhaseBatchResult(phase=PipelinePhase.VAD, status=PhaseStatus.COMPLETED,
                                                         items=[VadJobResult(source_id=source_id, success=True)])
        mock_domain_adaptation.execute.return_value = PhaseBatchResult(phase=PipelinePhase.DOMAIN_ADAPTATION,
                                                                       status=PhaseStatus.COMPLETED, items=[
                DomainAdaptationJobResult(source_id=source_id, success=True)])
        mock_transcribe.execute.return_value = PhaseBatchResult(phase=PipelinePhase.TRANSCRIBE,
                                                                status=PhaseStatus.COMPLETED, items=[
                TranscribeJobResult(source_id=source_id, success=True)])
        mock_diarize.execute.return_value = PhaseBatchResult(phase=PipelinePhase.DIARIZE, status=PhaseStatus.COMPLETED,
                                                             items=[
                                                                 DiarizeJobResult(source_id=source_id, success=True)])
        mock_sentence.execute.return_value = PhaseBatchResult(phase=PipelinePhase.SENTENCE_RECONSTRUCTION,
                                                              status=PhaseStatus.COMPLETED, items=[
                SentenceReconstructionJobResult(source_id=source_id, success=True,
                                                reconstructed_transcript_id=uuid.uuid4())])
        mock_phonetic.execute.return_value = PhaseBatchResult(phase=PipelinePhase.PHONETIC_CORRECTION,
                                                              status=PhaseStatus.COMPLETED, items=[
                PhoneticCorrectionJobResult(source_id=source_id, success=True, phonetic_transcript_id=uuid.uuid4())])

        # Execute
        results = pipeline.process_directory(target_dir, SourceType.VIDEO)

        # Assertions
        assert len(results) == 9

        # Assert Preloads triggered right before GPU bounds
        mock_vad.preload_model.assert_called_once()
        mock_transcribe.preload_model.assert_called_once()
        mock_diarize.preload_model.assert_called_once()
        mock_sentence.preload_model.assert_called_once()
        mock_phonetic.preload_model.assert_called_once()

        assert PipelinePhase.PHONETIC_CORRECTION in results
        assert results[PipelinePhase.PHONETIC_CORRECTION].status == PhaseStatus.COMPLETED