# Canvas: test_batch_media_pipeline_unit_v19
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
    HallucinationCorrectionJobResult,
    DiarizeJobResult,
    SentenceReconstructionJobResult,
    DomainPhoneticCorrectionJobResult,
    ContextualRetrievalJobResult,
    VisualCueDetectionJobResult,
    FrameExtractionJobResult,
    VisualAnalysisJobResult,
    VisualHallucinationCorrectionJobResult,
    VisualContextSynthesisJobResult,
    ContextUnificationJobResult,
    ClipExtractionJobResult,
    RagIndexingJobResult
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
    def mock_visual_cue(self): return Mock()

    @pytest.fixture
    def mock_frame_extraction(self): return Mock()

    @pytest.fixture
    def mock_visual_analysis(self): return Mock()

    @pytest.fixture
    def mock_visual_hallucination_correction(self): return Mock()

    @pytest.fixture
    def mock_visual_context_synthesis(self): return Mock()

    @pytest.fixture
    def mock_context_unification(self): return Mock()

    @pytest.fixture
    def mock_clip_extraction(self): return Mock()

    @pytest.fixture
    def mock_rag_indexing(self): return Mock()

    @pytest.fixture
    def pipeline(self,
                 mock_scan, mock_ingest, mock_extract, mock_vad,
                 mock_domain_adaptation, mock_transcribe, mock_hallucination_correction,
                 mock_diarize, mock_sentence, mock_phonetic, mock_contextual,
                 mock_visual_cue, mock_frame_extraction, mock_visual_analysis,
                 mock_visual_hallucination_correction, mock_visual_context_synthesis,
                 mock_context_unification, mock_clip_extraction, mock_rag_indexing):
        return BatchMediaPipeline(
            scan_phase=mock_scan,
            ingest_phase=mock_ingest,
            extract_audio_phase=mock_extract,
            vad_phase=mock_vad,
            domain_adaptation_phase=mock_domain_adaptation,
            transcribe_phase=mock_transcribe,
            hallucination_correction_phase=mock_hallucination_correction,
            diarize_phase=mock_diarize,
            sentence_reconstruction_phase=mock_sentence,
            domain_phonetic_correction_phase=mock_phonetic,
            contextual_retrieval_phase=mock_contextual,
            visual_cue_detection_phase=mock_visual_cue,
            frame_extraction_phase=mock_frame_extraction,
            visual_analysis_phase=mock_visual_analysis,
            visual_hallucination_correction_phase=mock_visual_hallucination_correction,
            visual_context_synthesis_phase=mock_visual_context_synthesis,
            context_unification_phase=mock_context_unification,
            clip_extraction_phase=mock_clip_extraction,
            rag_indexing_phase=mock_rag_indexing
        )

    @staticmethod
    def _configure_successful_run(source_id, mock_scan, mock_ingest, mock_extract, mock_vad,
                                  mock_domain_adaptation, mock_transcribe, mock_hallucination_correction,
                                  mock_diarize, mock_sentence, mock_phonetic, mock_contextual,
                                  mock_visual_cue, mock_frame_extraction, mock_visual_analysis,
                                  mock_visual_hallucination_correction, mock_visual_context_synthesis,
                                  mock_context_unification, mock_clip_extraction, mock_rag_indexing):
        """Every phase succeeds for one video source."""

        # Mock all 19 responses
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
        mock_hallucination_correction.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.HALLUCINATION_CORRECTION, status=PhaseStatus.COMPLETED,
            items=[HallucinationCorrectionJobResult(source_id=source_id, success=True, transcript_id=uuid.uuid4())])
        mock_diarize.execute.return_value = PhaseBatchResult(phase=PipelinePhase.DIARIZE, status=PhaseStatus.COMPLETED,
                                                             items=[
                                                                 DiarizeJobResult(source_id=source_id, success=True)])
        mock_sentence.execute.return_value = PhaseBatchResult(phase=PipelinePhase.SENTENCE_RECONSTRUCTION,
                                                              status=PhaseStatus.COMPLETED, items=[
                SentenceReconstructionJobResult(source_id=source_id, success=True,
                                                reconstructed_transcript_id=uuid.uuid4())])
        mock_phonetic.execute.return_value = PhaseBatchResult(phase=PipelinePhase.DOMAIN_PHONETIC_CORRECTION,
                                                              status=PhaseStatus.COMPLETED, items=[
                DomainPhoneticCorrectionJobResult(source_id=source_id, success=True,
                                                  domain_phonetic_transcript_id=uuid.uuid4())])
        mock_contextual.execute.return_value = PhaseBatchResult(phase=PipelinePhase.CONTEXTUAL_RETRIEVAL,
                                                                status=PhaseStatus.COMPLETED, items=[
                ContextualRetrievalJobResult(source_id=source_id, success=True, contextual_transcript_id=uuid.uuid4())])
        mock_visual_cue.execute.return_value = PhaseBatchResult(phase=PipelinePhase.VISUAL_CUE_DETECTION,
                                                                status=PhaseStatus.COMPLETED, items=[
                VisualCueDetectionJobResult(source_id=source_id, success=True, evaluated_transcript_id=uuid.uuid4())])
        mock_frame_extraction.execute.return_value = PhaseBatchResult(phase=PipelinePhase.FRAME_EXTRACTION,
                                                                      status=PhaseStatus.COMPLETED, items=[
                FrameExtractionJobResult(source_id=source_id, success=True, extracted_chunks_count=1)])
        mock_visual_analysis.execute.return_value = PhaseBatchResult(phase=PipelinePhase.VISUAL_ANALYSIS,
                                                                     status=PhaseStatus.COMPLETED, items=[
                VisualAnalysisJobResult(source_id=source_id, success=True, analyzed_chunks_count=1)])
        mock_visual_hallucination_correction.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.VISUAL_HALLUCINATION_CORRECTION, status=PhaseStatus.COMPLETED, items=[
                VisualHallucinationCorrectionJobResult(source_id=source_id, success=True,
                                                       evaluated_transcript_id=uuid.uuid4())])
        mock_visual_context_synthesis.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.VISUAL_CONTEXT_SYNTHESIS, status=PhaseStatus.COMPLETED,
            items=[VisualContextSynthesisJobResult(source_id=source_id, success=True, visual_timeline_id=uuid.uuid4())])
        mock_context_unification.execute.return_value = PhaseBatchResult(phase=PipelinePhase.CONTEXT_UNIFICATION,
                                                                         status=PhaseStatus.COMPLETED, items=[
                ContextUnificationJobResult(source_id=source_id, success=True, unified_transcript_id=uuid.uuid4())])

        mock_clip_extraction.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.CLIP_EXTRACTION,
            status=PhaseStatus.COMPLETED,
            items=[ClipExtractionJobResult(source_id=source_id, success=True, extracted_clips_count=2)]
        )

        mock_rag_indexing.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.RAG_INDEXING,
            status=PhaseStatus.COMPLETED,
            items=[
                RagIndexingJobResult(source_id=source_id, success=True, dense_vectors_saved=5, sparse_vectors_saved=5)]
        )

    def test_pipeline_happy_path(self, pipeline, mock_scan, mock_ingest, mock_extract, mock_vad,
                                 mock_domain_adaptation, mock_transcribe, mock_hallucination_correction,
                                 mock_diarize, mock_sentence, mock_phonetic, mock_contextual,
                                 mock_visual_cue, mock_frame_extraction, mock_visual_analysis,
                                 mock_visual_hallucination_correction, mock_visual_context_synthesis,
                                 mock_context_unification, mock_clip_extraction, mock_rag_indexing):
        target_dir = "/dummy/path"
        source_id = uuid.uuid4()
        self._configure_successful_run(source_id, mock_scan, mock_ingest, mock_extract, mock_vad, mock_domain_adaptation, mock_transcribe, mock_hallucination_correction, mock_diarize, mock_sentence, mock_phonetic, mock_contextual, mock_visual_cue, mock_frame_extraction, mock_visual_analysis, mock_visual_hallucination_correction, mock_visual_context_synthesis, mock_context_unification, mock_clip_extraction, mock_rag_indexing)

        results = pipeline.process_directory(target_dir, SourceType.VIDEO)

        assert len(results) == 19  # All 19 phases

        assert PipelinePhase.CLIP_EXTRACTION in results
        assert PipelinePhase.RAG_INDEXING in results

        mock_clip_extraction.execute.assert_called_once()
        mock_rag_indexing.execute.assert_called_once()

    def test_preloads_left_by_a_halted_visual_branch_are_released_before_the_embedder_preload(
            self, pipeline, mock_scan, mock_ingest, mock_extract, mock_vad, mock_domain_adaptation, mock_transcribe, mock_hallucination_correction, mock_diarize, mock_sentence, mock_phonetic, mock_contextual, mock_visual_cue, mock_frame_extraction, mock_visual_analysis, mock_visual_hallucination_correction, mock_visual_context_synthesis, mock_context_unification, mock_clip_extraction, mock_rag_indexing):
        """
        The synthesis preload is triggered before visual hallucination correction runs. When that phase fails, the
        synthesis model loads with no phase left to release it, so it must be discarded before the embedder preload
        queues behind it (the preload wait has no limit).
        """
        source_id = uuid.uuid4()
        self._configure_successful_run(source_id, mock_scan, mock_ingest, mock_extract, mock_vad, mock_domain_adaptation, mock_transcribe, mock_hallucination_correction, mock_diarize, mock_sentence, mock_phonetic, mock_contextual, mock_visual_cue, mock_frame_extraction, mock_visual_analysis, mock_visual_hallucination_correction, mock_visual_context_synthesis, mock_context_unification, mock_clip_extraction, mock_rag_indexing)
        mock_visual_hallucination_correction.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.VISUAL_HALLUCINATION_CORRECTION, status=PhaseStatus.FAILED, items=[])
        events = []
        mock_visual_context_synthesis.discard_preload.side_effect = lambda: events.append("discard synthesis")
        mock_rag_indexing.preload_model.side_effect = lambda: events.append("preload embedder")

        pipeline.process_directory("/dummy/path", SourceType.VIDEO)

        mock_visual_context_synthesis.preload_model.assert_called_once()
        mock_visual_context_synthesis.execute.assert_not_called()
        assert events[:2] == ["discard synthesis", "preload embedder"]
        mock_rag_indexing.execute.assert_called_once()
