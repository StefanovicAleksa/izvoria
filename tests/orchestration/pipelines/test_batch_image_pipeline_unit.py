# Path: tests/orchestration/pipelines/test_batch_image_pipeline_unit.py

import uuid
from unittest.mock import Mock

import pytest

from app.features.storage.domain.entities import SourceType
from app.orchestration.domain.entities import (
    ContextUnificationJobResult, ImageDescriptionJobResult, IngestJobResult, PhaseBatchResult, RagIndexingJobResult,
)
from app.orchestration.domain.enums import PhaseStatus, PipelinePhase
from app.orchestration.exceptions.workflow import WorkflowExecutionError
from app.orchestration.service.pipelines.batch_image_pipeline import BatchImagePipeline

PHASE_NAMES = ["scan", "ingest", "description", "unification", "rag_indexing"]


def _result(phase, *items, status=PhaseStatus.COMPLETED):
    return PhaseBatchResult(phase=phase, status=status, successful_count=len(items), items=list(items))


class TestBatchImagePipeline:
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
        return BatchImagePipeline(
            scan_phase=phases["scan"], ingest_phase=phases["ingest"], image_description_phase=phases["description"],
            context_unification_phase=phases["unification"], rag_indexing_phase=phases["rag_indexing"],
        )

    def _happy(self, phases, source_id):
        phases["scan"].execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.SCAN, status=PhaseStatus.COMPLETED, successful_count=5,
            items=["/in/board.JPG", "/in/scan.tiff", "/in/chart.png", "/in/lease.pdf", "/in/talk.mp4"])
        phases["ingest"].execute.return_value = _result(
            PipelinePhase.INGEST, IngestJobResult(file_path="/in/board.JPG", success=True, source_id=source_id))
        described = ImageDescriptionJobResult(source_id=source_id, success=True, ocr_checked=True)
        phases["description"].execute.return_value = _result(PipelinePhase.IMAGE_DESCRIPTION, described)
        unified = ContextUnificationJobResult(source_id=source_id, success=True)
        phases["unification"].execute.return_value = _result(PipelinePhase.CONTEXT_UNIFICATION, unified)
        phases["rag_indexing"].execute.return_value = _result(
            PipelinePhase.RAG_INDEXING, RagIndexingJobResult(source_id=source_id, success=True))
        return described, unified

    def test_ingests_only_images_and_runs_every_phase_in_order(self, pipeline, phases):
        source_id = uuid.uuid4()
        described, unified = self._happy(phases, source_id)

        results = pipeline.process_directory("/in")

        phases["ingest"].execute.assert_called_once_with(
            ["/in/board.JPG", "/in/scan.tiff", "/in/chart.png"], SourceType.IMAGE)
        phases["description"].execute.assert_called_once_with([source_id])
        phases["unification"].execute.assert_called_once_with([described])
        phases["rag_indexing"].execute.assert_called_once_with([unified])
        assert set(results) == {PipelinePhase.SCAN, PipelinePhase.INGEST, PipelinePhase.IMAGE_DESCRIPTION,
                                PipelinePhase.CONTEXT_UNIFICATION, PipelinePhase.RAG_INDEXING}

    def test_vlm_preloads_while_ingest_copies_files_and_embedder_before_unification(self, pipeline, phases):
        self._happy(phases, uuid.uuid4())

        pipeline.process_directory("/in")

        order = [name for name, _, _ in phases["manager"].mock_calls if name.endswith(("preload_model", ".execute"))]
        assert order == [
            "scan.execute", "description.preload_model", "ingest.execute", "description.execute",
            "rag_indexing.preload_model", "unification.execute", "rag_indexing.execute",
        ]

    def test_no_images_stops_before_any_model_loads(self, pipeline, phases):
        phases["scan"].execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.SCAN, status=PhaseStatus.COMPLETED, successful_count=1, items=["/in/a.pdf"])

        results = pipeline.process_directory("/in")

        phases["ingest"].execute.assert_not_called()
        phases["description"].preload_model.assert_not_called()
        assert list(results) == [PipelinePhase.SCAN]

    def test_nothing_ingested_discards_the_vlm_preload(self, pipeline, phases):
        self._happy(phases, uuid.uuid4())
        phases["ingest"].execute.return_value = _result(
            PipelinePhase.INGEST, IngestJobResult(file_path="/in/board.JPG", success=False))

        pipeline.process_directory("/in")

        phases["description"].execute.assert_not_called()
        phases["description"].discard_preload.assert_called_once_with()
        phases["rag_indexing"].discard_preload.assert_called_once_with()

    def test_every_image_failing_skips_indexing(self, pipeline, phases):
        source_id = uuid.uuid4()
        self._happy(phases, source_id)
        phases["description"].execute.return_value = _result(
            PipelinePhase.IMAGE_DESCRIPTION, ImageDescriptionJobResult(source_id=source_id, success=False),
            status=PhaseStatus.FAILED)

        results = pipeline.process_directory("/in")

        phases["rag_indexing"].preload_model.assert_not_called()
        phases["unification"].execute.assert_not_called()
        assert PipelinePhase.IMAGE_DESCRIPTION in results

    def test_a_crash_is_wrapped_and_preloads_are_discarded(self, pipeline, phases):
        self._happy(phases, uuid.uuid4())
        phases["description"].execute.side_effect = RuntimeError("boom")

        with pytest.raises(WorkflowExecutionError):
            pipeline.process_directory("/in")

        phases["description"].discard_preload.assert_called_once_with()
        phases["rag_indexing"].discard_preload.assert_called_once_with()
