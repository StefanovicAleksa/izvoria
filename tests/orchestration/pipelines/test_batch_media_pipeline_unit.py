# Canvas: test_batch_media_pipeline_unit_v12
# Path: tests/orchestration/pipelines/test_batch_media_pipeline_unit.py

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
    DomainPhoneticCorrectionJobResult,
    ContextualRetrievalJobResult,
    VisualCueDetectionJobResult,
    FrameExtractionJobResult,
    VisualAnalysisJobResult
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
    def mock_contextual(self): return Mock()
    @pytest.fixture
    def mock_visual_cue(self): return Mock()
    @pytest.fixture
    def mock_frame_extraction(self): return Mock()
    @pytest.fixture
    def mock_visual_analysis(self): return Mock()

    @pytest.fixture
    def pipeline(self, mock_scan, mock_ingest, mock_extract, mock_vad, mock_domain_adaptation, mock_transcribe, mock_diarize, mock_sentence, mock_phonetic, mock_contextual, mock_visual_cue, mock_frame_extraction, mock_visual_analysis):
        return BatchMediaPipeline(
            scan_phase=mock_scan,
            ingest_phase=mock_ingest,
            extract_audio_phase=mock_extract,
            vad_phase=mock_vad,
            domain_adaptation_phase=mock_domain_adaptation,
            transcribe_phase=mock_transcribe,
            diarize_phase=mock_diarize,
            sentence_reconstruction_phase=mock_sentence,
            domain_phonetic_correction_phase=mock_phonetic,
            contextual_retrieval_phase=mock_contextual,
            visual_cue_detection_phase=mock_visual_cue,
            frame_extraction_phase=mock_frame_extraction,
            visual_analysis_phase=mock_visual_analysis
        )

    def test_pipeline_happy_path(self, pipeline, mock_scan, mock_ingest, mock_extract, mock_vad, mock_domain_adaptation, mock_transcribe, mock_diarize, mock_sentence, mock_phonetic, mock_contextual, mock_visual_cue, mock_frame_extraction, mock_visual_analysis):
        target_dir = "/dummy/path"
        source_id = uuid.uuid4()

        mock_scan.execute.return_value = PhaseBatchResult(phase=PipelinePhase.SCAN, status=PhaseStatus.COMPLETED, successful_count=1, items=["/dummy/path/vid.mp4"])
        mock_ingest.execute.return_value = PhaseBatchResult(phase=PipelinePhase.INGEST, status=PhaseStatus.COMPLETED, items=[IngestJobResult(source_id=source_id, success=True, file_path="/dummy/path/vid.mp4")])
        mock_extract.execute.return_value = PhaseBatchResult(phase=PipelinePhase.AUDIO_EXTRACTION, status=PhaseStatus.COMPLETED, items=[AudioExtractionJobResult(source_id=source_id, success=True)])
        mock_vad.execute.return_value = PhaseBatchResult(phase=PipelinePhase.VAD, status=PhaseStatus.COMPLETED, items=[VadJobResult(source_id=source_id, success=True)])
        mock_domain_adaptation.execute.return_value = PhaseBatchResult(phase=PipelinePhase.DOMAIN_ADAPTATION, status=PhaseStatus.COMPLETED, items=[DomainAdaptationJobResult(source_id=source_id, success=True)])
        mock_transcribe.execute.return_value = PhaseBatchResult(phase=PipelinePhase.TRANSCRIBE, status=PhaseStatus.COMPLETED, items=[TranscribeJobResult(source_id=source_id, success=True)])
        mock_diarize.execute.return_value = PhaseBatchResult(phase=PipelinePhase.DIARIZE, status=PhaseStatus.COMPLETED, items=[DiarizeJobResult(source_id=source_id, success=True)])
        mock_sentence.execute.return_value = PhaseBatchResult(phase=PipelinePhase.SENTENCE_RECONSTRUCTION, status=PhaseStatus.COMPLETED, items=[SentenceReconstructionJobResult(source_id=source_id, success=True, reconstructed_transcript_id=uuid.uuid4())])
        mock_phonetic.execute.return_value = PhaseBatchResult(phase=PipelinePhase.DOMAIN_PHONETIC_CORRECTION, status=PhaseStatus.COMPLETED, items=[DomainPhoneticCorrectionJobResult(source_id=source_id, success=True, domain_phonetic_transcript_id=uuid.uuid4())])
        mock_contextual.execute.return_value = PhaseBatchResult(phase=PipelinePhase.CONTEXTUAL_RETRIEVAL, status=PhaseStatus.COMPLETED, items=[ContextualRetrievalJobResult(source_id=source_id, success=True, contextual_transcript_id=uuid.uuid4())])
        mock_visual_cue.execute.return_value = PhaseBatchResult(phase=PipelinePhase.VISUAL_CUE_DETECTION, status=PhaseStatus.COMPLETED, items=[VisualCueDetectionJobResult(source_id=source_id, success=True, evaluated_transcript_id=uuid.uuid4())])
        mock_frame_extraction.execute.return_value = PhaseBatchResult(phase=PipelinePhase.FRAME_EXTRACTION, status=PhaseStatus.COMPLETED, items=[FrameExtractionJobResult(source_id=source_id, success=True, extracted_chunks_count=1)])
        mock_visual_analysis.execute.return_value = PhaseBatchResult(phase=PipelinePhase.VISUAL_ANALYSIS, status=PhaseStatus.COMPLETED, items=[VisualAnalysisJobResult(source_id=source_id, success=True, analyzed_chunks_count=1)])

        results = pipeline.process_directory(target_dir, SourceType.VIDEO)

        assert len(results) == 13
        mock_phonetic.preload_model.assert_called_once()
        assert PipelinePhase.VISUAL_ANALYSIS in results