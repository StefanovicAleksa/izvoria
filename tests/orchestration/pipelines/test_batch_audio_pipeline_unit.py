# Canvas: test_batch_audio_pipeline_unit_v1
# Path: tests/orchestration/pipelines/test_batch_audio_pipeline_unit.py

import pytest
import uuid
from unittest.mock import Mock

from app.orchestration.service.pipelines.batch_audio_pipeline import BatchAudioPipeline
from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import (
    PhaseBatchResult,
    IngestJobResult,
    VadJobResult,
    DomainAdaptationJobResult,
    TranscribeJobResult,
    HallucinationCorrectionJobResult,
    DiarizeJobResult,
    SentenceReconstructionJobResult,
    DomainPhoneticCorrectionJobResult,
    ContextualRetrievalJobResult,
    ContextUnificationJobResult,
    RagIndexingJobResult
)
from app.features.storage.domain.entities import SourceType, Source


class TestBatchAudioPipelineUnit:
    @pytest.fixture
    def mock_scan(self): return Mock()

    @pytest.fixture
    def mock_ingest(self): return Mock()

    @pytest.fixture
    def mock_vad(self): return Mock()

    @pytest.fixture
    def mock_domain_adaptation(self): return Mock()

    @pytest.fixture
    def mock_transcribe(self): return Mock()

    @pytest.fixture
    def mock_hallucination_correction(self): return Mock()

    @pytest.fixture
    def mock_diarize(self): return Mock()

    @pytest.fixture
    def mock_sentence(self): return Mock()

    @pytest.fixture
    def mock_phonetic(self): return Mock()

    @pytest.fixture
    def mock_contextual(self): return Mock()

    @pytest.fixture
    def mock_context_unification(self): return Mock()

    @pytest.fixture
    def mock_rag_indexing(self): return Mock()

    @pytest.fixture
    def mock_storage_api(self): return Mock()

    @pytest.fixture
    def pipeline(self,
                 mock_scan, mock_ingest, mock_vad,
                 mock_domain_adaptation, mock_transcribe, mock_hallucination_correction,
                 mock_diarize, mock_sentence, mock_phonetic, mock_contextual,
                 mock_context_unification, mock_rag_indexing, mock_storage_api):
        return BatchAudioPipeline(
            scan_phase=mock_scan,
            ingest_phase=mock_ingest,
            vad_phase=mock_vad,
            domain_adaptation_phase=mock_domain_adaptation,
            transcribe_phase=mock_transcribe,
            hallucination_correction_phase=mock_hallucination_correction,
            diarize_phase=mock_diarize,
            sentence_reconstruction_phase=mock_sentence,
            domain_phonetic_correction_phase=mock_phonetic,
            contextual_retrieval_phase=mock_contextual,
            context_unification_phase=mock_context_unification,
            rag_indexing_phase=mock_rag_indexing,
            storage_api=mock_storage_api
        )

    def test_pipeline_happy_path_and_filtering(self, pipeline, mock_scan, mock_ingest, mock_vad,
                                               mock_domain_adaptation, mock_transcribe, mock_hallucination_correction,
                                               mock_diarize, mock_sentence, mock_phonetic, mock_contextual,
                                               mock_context_unification, mock_rag_indexing, mock_storage_api):
        target_dir = "/dummy/path"
        source_id = uuid.uuid4()
        file_id = uuid.uuid4()

        # Mock Scan to return both an audio and a non-audio file to test filtering
        mock_scan.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.SCAN, status=PhaseStatus.COMPLETED, successful_count=2,
            items=["/dummy/path/audio.mp3", "/dummy/path/ignore.txt"]
        )

        mock_ingest.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.INGEST, status=PhaseStatus.COMPLETED,
            items=[IngestJobResult(source_id=source_id, success=True, file_path="/dummy/path/audio.mp3")]
        )

        # Mock Storage API for the Data Bridge
        mock_source = Source(id=source_id, name="audio", source_type=SourceType.AUDIO, content_hash="123",
                             file_id=file_id)
        mock_storage_api.get_source.return_value = mock_source

        # Mock the rest of the phases
        mock_vad.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.VAD, status=PhaseStatus.COMPLETED,
            items=[VadJobResult(source_id=source_id, success=True)]
        )
        mock_domain_adaptation.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.DOMAIN_ADAPTATION, status=PhaseStatus.COMPLETED,
            items=[DomainAdaptationJobResult(source_id=source_id, success=True)]
        )
        mock_transcribe.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.TRANSCRIBE, status=PhaseStatus.COMPLETED,
            items=[TranscribeJobResult(source_id=source_id, success=True)]
        )
        mock_hallucination_correction.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.HALLUCINATION_CORRECTION, status=PhaseStatus.COMPLETED,
            items=[HallucinationCorrectionJobResult(source_id=source_id, success=True, transcript_id=uuid.uuid4())]
        )
        mock_diarize.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.DIARIZE, status=PhaseStatus.COMPLETED,
            items=[DiarizeJobResult(source_id=source_id, success=True)]
        )
        mock_sentence.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.SENTENCE_RECONSTRUCTION, status=PhaseStatus.COMPLETED, items=[
                SentenceReconstructionJobResult(source_id=source_id, success=True,
                                                reconstructed_transcript_id=uuid.uuid4())]
        )
        mock_phonetic.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.DOMAIN_PHONETIC_CORRECTION, status=PhaseStatus.COMPLETED, items=[
                DomainPhoneticCorrectionJobResult(source_id=source_id, success=True,
                                                  domain_phonetic_transcript_id=uuid.uuid4())]
        )
        mock_contextual.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.CONTEXTUAL_RETRIEVAL, status=PhaseStatus.COMPLETED, items=[
                ContextualRetrievalJobResult(source_id=source_id, success=True, contextual_transcript_id=uuid.uuid4())]
        )
        mock_context_unification.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.CONTEXT_UNIFICATION, status=PhaseStatus.COMPLETED,
            items=[ContextUnificationJobResult(source_id=source_id, success=True, unified_transcript_id=uuid.uuid4())]
        )
        mock_rag_indexing.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.RAG_INDEXING, status=PhaseStatus.COMPLETED, items=[
                RagIndexingJobResult(source_id=source_id, success=True, dense_vectors_saved=5, sparse_vectors_saved=5)]
        )

        results = pipeline.process_directory(target_dir)

        # Assert correct filtering during ingest
        mock_ingest.execute.assert_called_once_with(["/dummy/path/audio.mp3"], SourceType.AUDIO)

        # Ensure we processed 12 specific phases (Audio bypasses all 7 visual/video phases)
        assert len(results) == 12

        assert PipelinePhase.AUDIO_EXTRACTION not in results
        assert PipelinePhase.VISUAL_ANALYSIS not in results
        assert PipelinePhase.CLIP_EXTRACTION not in results

        assert PipelinePhase.SCAN in results
        assert PipelinePhase.INGEST in results
        assert PipelinePhase.VAD in results
        assert PipelinePhase.RAG_INDEXING in results

        mock_rag_indexing.execute.assert_called_once()