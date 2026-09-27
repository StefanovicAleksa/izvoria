# Path: tests/orchestration/pipelines/test_batch_document_pipeline_unit.py

import uuid
from unittest.mock import Mock

import pytest

from app.features.storage.domain.entities import SourceType
from app.orchestration.domain.entities import (
    ContextualRetrievalJobResult, ContextUnificationJobResult, DocumentConversionJobResult, IngestJobResult,
    PhaseBatchResult, RagIndexingJobResult,
)
from app.orchestration.domain.enums import PhaseStatus, PipelinePhase
from app.orchestration.exceptions.workflow import WorkflowStateError
from app.orchestration.service.pipelines.batch_document_pipeline import BatchDocumentPipeline

PHASE_NAMES = ["scan", "ingest", "conversion", "contextualization", "unification", "rag_indexing"]


def _result(phase, *items, status=PhaseStatus.COMPLETED):
    return PhaseBatchResult(phase=phase, status=status, successful_count=len(items), items=list(items))


class TestBatchDocumentPipeline:
    @pytest.fixture
    def phases(self):
        phases = {name: Mock() for name in PHASE_NAMES}
        manager = Mock()
        for name, phase in phases.items():
            manager.attach_mock(phase, name)
        phases["manager"] = manager
        return phases

    @pytest.fixture
    def pipeline(self, phases):
        return BatchDocumentPipeline(
            scan_phase=phases["scan"], ingest_phase=phases["ingest"], document_conversion_phase=phases["conversion"],
            document_contextualization_phase=phases["contextualization"],
            context_unification_phase=phases["unification"], rag_indexing_phase=phases["rag_indexing"],
        )

    def _happy(self, phases, source_id):
        phases["scan"].execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.SCAN, status=PhaseStatus.COMPLETED, successful_count=3,
            items=["/in/lease.PDF", "/in/memo.docx", "/in/interview.mp3"])
        phases["ingest"].execute.return_value = _result(
            PipelinePhase.INGEST, IngestJobResult(file_path="/in/lease.PDF", success=True, source_id=source_id))
        conversion = DocumentConversionJobResult(source_id=source_id, success=True, pages_total=4)
        phases["conversion"].execute.return_value = _result(PipelinePhase.DOCUMENT_CONVERSION, conversion)
        contextual = ContextualRetrievalJobResult(source_id=source_id, success=True)
        phases["contextualization"].execute.return_value = _result(PipelinePhase.DOCUMENT_CONTEXTUALIZATION,
                                                                   contextual)
        unified = ContextUnificationJobResult(source_id=source_id, success=True)
        phases["unification"].execute.return_value = _result(PipelinePhase.CONTEXT_UNIFICATION, unified)
        phases["rag_indexing"].execute.return_value = _result(
            PipelinePhase.RAG_INDEXING, RagIndexingJobResult(source_id=source_id, success=True))
        return conversion, contextual, unified

    def test_ingests_only_documents_and_runs_every_phase_in_order(self, pipeline, phases):
        source_id = uuid.uuid4()
        conversion, contextual, unified = self._happy(phases, source_id)

        results = pipeline.process_directory("/in")

        phases["ingest"].execute.assert_called_once_with(["/in/lease.PDF", "/in/memo.docx"], SourceType.DOCUMENT)
        phases["conversion"].execute.assert_called_once_with([source_id])
        phases["contextualization"].execute.assert_called_once_with([conversion])
        phases["unification"].execute.assert_called_once_with([contextual])
        phases["rag_indexing"].execute.assert_called_once_with([unified])
        assert set(results) == {PipelinePhase.SCAN, PipelinePhase.INGEST, PipelinePhase.DOCUMENT_CONVERSION,
                                PipelinePhase.DOCUMENT_CONTEXTUALIZATION, PipelinePhase.CONTEXT_UNIFICATION,
                                PipelinePhase.RAG_INDEXING}

    def test_slm_preloads_while_conversion_runs_and_embedder_before_unification(self, pipeline, phases):
        self._happy(phases, uuid.uuid4())

        pipeline.process_directory("/in")

        order = [name for name, _, _ in phases["manager"].mock_calls if name.endswith(("preload_model", ".execute"))]
        assert order[order.index("contextualization.preload_model"):] == [
            "contextualization.preload_model", "conversion.execute", "contextualization.execute",
            "rag_indexing.preload_model", "unification.execute", "rag_indexing.execute",
        ]

    def test_books_and_texts_are_ingested_as_documents(self, pipeline, phases):
        source_id = uuid.uuid4()
        self._happy(phases, source_id)
        phases["scan"].execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.SCAN, status=PhaseStatus.COMPLETED, successful_count=7,
            items=["/in/novel.txt", "/in/notes.md", "/in/book.EPUB", "/in/page.html", "/in/letter.rtf",
                   "/in/data.csv", "/in/talk.mp3"])

        pipeline.process_directory("/in")

        phases["ingest"].execute.assert_called_once_with(
            ["/in/novel.txt", "/in/notes.md", "/in/book.EPUB", "/in/page.html", "/in/letter.rtf"],
            SourceType.DOCUMENT)

    def test_no_documents_stops_before_ingest(self, pipeline, phases):
        phases["scan"].execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.SCAN, status=PhaseStatus.COMPLETED, successful_count=1, items=["/in/a.mp4"])

        results = pipeline.process_directory("/in")

        phases["ingest"].execute.assert_not_called()
        assert list(results) == [PipelinePhase.SCAN]

    def test_conversion_failing_for_every_file_halts_and_discards_the_preload(self, pipeline, phases):
        source_id = uuid.uuid4()
        self._happy(phases, source_id)
        phases["conversion"].execute.return_value = _result(
            PipelinePhase.DOCUMENT_CONVERSION, DocumentConversionJobResult(source_id=source_id, success=False),
            status=PhaseStatus.FAILED)

        with pytest.raises(WorkflowStateError):
            pipeline.process_directory("/in")

        phases["contextualization"].execute.assert_not_called()
        phases["contextualization"].discard_preload.assert_called_once_with()
        phases["rag_indexing"].discard_preload.assert_called_once_with()
