# Canvas: test_batch_audio_pipeline_unit_v2
# Path: tests/orchestration/pipelines/test_batch_audio_pipeline_unit.py

import pytest
import uuid
from unittest.mock import Mock

from app.orchestration.service.pipelines.batch_audio_pipeline import BatchAudioPipeline
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
    ContextUnificationJobResult,
    RagIndexingJobResult
)
from app.features.storage.domain.entities import SourceType

PHASE_NAMES = [
    "scan", "ingest", "extract_audio", "vad", "domain_adaptation", "transcribe", "hallucination_correction",
    "diarize", "sentence", "phonetic", "contextual", "context_unification", "rag_indexing"
]


class TestBatchAudioPipelineUnit:
    @pytest.fixture
    def phases(self):
        phases = {name: Mock() for name in PHASE_NAMES}
        phases["storage_compaction"] = Mock()
        phases["storage_compaction"].enabled.return_value = False  # the appliance's defaults
        return phases

    @pytest.fixture
    def pipeline(self, phases):
        return BatchAudioPipeline(
            scan_phase=phases["scan"],
            ingest_phase=phases["ingest"],
            extract_audio_phase=phases["extract_audio"],
            vad_phase=phases["vad"],
            domain_adaptation_phase=phases["domain_adaptation"],
            transcribe_phase=phases["transcribe"],
            hallucination_correction_phase=phases["hallucination_correction"],
            diarize_phase=phases["diarize"],
            sentence_reconstruction_phase=phases["sentence"],
            domain_phonetic_correction_phase=phases["phonetic"],
            contextual_retrieval_phase=phases["contextual"],
            context_unification_phase=phases["context_unification"],
            rag_indexing_phase=phases["rag_indexing"],
            storage_compaction_phase=phases["storage_compaction"]
        )

    @staticmethod
    def _result(phase: PipelinePhase, item) -> PhaseBatchResult:
        return PhaseBatchResult(phase=phase, status=PhaseStatus.COMPLETED, successful_count=1, items=[item])

    def _configure_full_run(self, phases):
        source_id = uuid.uuid4()
        extracted_file_id = uuid.uuid4()

        phases["scan"].execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.SCAN, status=PhaseStatus.COMPLETED, successful_count=2,
            items=["/dummy/path/audio.mp3", "/dummy/path/ignore.txt"]
        )
        phases["ingest"].execute.return_value = self._result(
            PipelinePhase.INGEST, IngestJobResult(source_id=source_id, success=True, file_path="/dummy/path/audio.mp3"))
        extraction_item = AudioExtractionJobResult(source_id=source_id, success=True, extracted_file_id=extracted_file_id)
        phases["extract_audio"].execute.return_value = self._result(PipelinePhase.AUDIO_EXTRACTION, extraction_item)
        vad_item = VadJobResult(source_id=source_id, success=True)
        phases["vad"].execute.return_value = self._result(PipelinePhase.VAD, vad_item)
        phases["domain_adaptation"].execute.return_value = self._result(
            PipelinePhase.DOMAIN_ADAPTATION, DomainAdaptationJobResult(source_id=source_id, success=True))
        phases["transcribe"].execute.return_value = self._result(
            PipelinePhase.TRANSCRIBE, TranscribeJobResult(source_id=source_id, success=True))
        phases["hallucination_correction"].execute.return_value = self._result(
            PipelinePhase.HALLUCINATION_CORRECTION,
            HallucinationCorrectionJobResult(source_id=source_id, success=True, transcript_id=uuid.uuid4()))
        phases["diarize"].execute.return_value = self._result(
            PipelinePhase.DIARIZE, DiarizeJobResult(source_id=source_id, success=True))
        phases["sentence"].execute.return_value = self._result(
            PipelinePhase.SENTENCE_RECONSTRUCTION,
            SentenceReconstructionJobResult(source_id=source_id, success=True, reconstructed_transcript_id=uuid.uuid4()))
        phases["phonetic"].execute.return_value = self._result(
            PipelinePhase.DOMAIN_PHONETIC_CORRECTION,
            DomainPhoneticCorrectionJobResult(source_id=source_id, success=True, domain_phonetic_transcript_id=uuid.uuid4()))
        phases["contextual"].execute.return_value = self._result(
            PipelinePhase.CONTEXTUAL_RETRIEVAL,
            ContextualRetrievalJobResult(source_id=source_id, success=True, contextual_transcript_id=uuid.uuid4()))
        phases["context_unification"].execute.return_value = self._result(
            PipelinePhase.CONTEXT_UNIFICATION,
            ContextUnificationJobResult(source_id=source_id, success=True, unified_transcript_id=uuid.uuid4()))
        phases["rag_indexing"].execute.return_value = self._result(
            PipelinePhase.RAG_INDEXING,
            RagIndexingJobResult(source_id=source_id, success=True, dense_vectors_saved=5, sparse_vectors_saved=5))
        return source_id, extraction_item, vad_item

    def test_pipeline_normalizes_audio_before_vad_and_filters_non_audio(self, pipeline, phases):
        source_id, extraction_item, vad_item = self._configure_full_run(phases)

        results = pipeline.process_directory("/dummy/path")

        phases["ingest"].execute.assert_called_once_with(["/dummy/path/audio.mp3"], SourceType.AUDIO)
        phases["extract_audio"].execute.assert_called_once_with([source_id])
        phases["vad"].preload_model.assert_called_once_with()
        phases["vad"].execute.assert_called_once_with([extraction_item])
        phases["domain_adaptation"].execute.assert_called_once_with([vad_item])

        assert len(results) == 13
        assert PipelinePhase.AUDIO_EXTRACTION in results
        assert PipelinePhase.VISUAL_ANALYSIS not in results
        assert PipelinePhase.CLIP_EXTRACTION not in results
        assert PipelinePhase.RAG_INDEXING in results

        for name in ("vad", "transcribe", "diarize", "sentence", "phonetic", "contextual", "rag_indexing"):
            phases[name].discard_preload.assert_called_once_with()

    def test_pipeline_stops_when_no_audio_could_be_normalized(self, pipeline, phases):
        source_id = uuid.uuid4()
        phases["scan"].execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.SCAN, status=PhaseStatus.COMPLETED, successful_count=1, items=["/d/a.wav"])
        phases["ingest"].execute.return_value = self._result(
            PipelinePhase.INGEST, IngestJobResult(source_id=source_id, success=True, file_path="/d/a.wav"))
        phases["extract_audio"].execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.AUDIO_EXTRACTION, status=PhaseStatus.PARTIAL_SUCCESS,
            items=[AudioExtractionJobResult(source_id=source_id, success=False, error_message="corrupt")])

        results = pipeline.process_directory("/d")

        assert PipelinePhase.VAD not in results
        phases["vad"].execute.assert_not_called()
        phases["vad"].discard_preload.assert_called_once_with()

    def test_storage_compaction_runs_last_on_what_was_indexed(self, pipeline, phases):
        source_id, _, _ = self._configure_full_run(phases)
        phases["storage_compaction"].enabled.return_value = True
        compacted = PhaseBatchResult(phase=PipelinePhase.STORAGE_COMPACTION, status=PhaseStatus.COMPLETED)
        phases["storage_compaction"].execute.return_value = compacted

        results = pipeline.process_directory("/dummy/path")

        indexed = phases["rag_indexing"].execute.return_value.items
        phases["storage_compaction"].execute.assert_called_once_with(indexed)
        assert indexed[0].source_id == source_id
        assert results[PipelinePhase.STORAGE_COMPACTION] is compacted

    def test_storage_compaction_is_skipped_when_disabled(self, pipeline, phases):
        self._configure_full_run(phases)

        results = pipeline.process_directory("/dummy/path")

        phases["storage_compaction"].execute.assert_not_called()
        assert PipelinePhase.STORAGE_COMPACTION not in results
