# Path: tests/orchestration/phases/test_document_contextualization_phase_unit.py

import uuid
from unittest.mock import Mock

from app.core.jobs.domain.enums import JobType
from app.core.model_manager.domain.enums import ModelType
from app.core.model_manager.exceptions.base import ModelManagerError
from app.orchestration.domain.entities import DocumentConversionJobResult
from app.orchestration.domain.enums import PhaseStatus, PipelinePhase
from app.orchestration.service.phases.document_contextualization_phase import DocumentContextualizationPhase


def _phase():
    deps = {name: Mock() for name in ("job_manager", "structuring", "contextualizer", "repo", "async", "loader")}
    deps["job_manager"].run_job.side_effect = lambda request, func, **kwargs: func(**kwargs)
    phase = DocumentContextualizationPhase(
        job_manager=deps["job_manager"], structuring_api=deps["structuring"],
        contextualizer_api=deps["contextualizer"], conversion_repo=deps["repo"],
        async_model_service=deps["async"], loader=deps["loader"])
    return phase, deps


class TestDocumentContextualizationPhase:
    def test_structures_then_contextualizes_with_the_model_held_for_the_batch(self):
        phase, deps = _phase()
        source_id = uuid.uuid4()
        deps["contextualizer"].execute_document.return_value.id = uuid.uuid4()
        phase.preload_model()

        result = phase.execute([DocumentConversionJobResult(source_id=source_id, success=True),
                                DocumentConversionJobResult(source_id=uuid.uuid4(), success=False)])

        manifest = deps["async"].trigger_preload.call_args.kwargs["manifest"]
        assert manifest.model_type is ModelType.CONTEXTUAL_RETRIEVAL
        document = deps["repo"].get_by_source_id.return_value
        deps["structuring"].execute.assert_called_once_with(document)
        deps["contextualizer"].execute_document.assert_called_once_with(
            deps["structuring"].execute.return_value, release_after=False)
        request = deps["job_manager"].run_job.call_args.args[0]
        assert (request.job_type, request.source_id) == (JobType.DOCUMENT_CONTEXTUALIZATION, source_id)
        deps["async"].trigger_unload.assert_called_once_with(
            loaded_model=deps["async"].resolve_preload.return_value, loader=deps["loader"], flush=True)
        assert (result.phase, result.status, result.successful_count) == (
            PipelinePhase.DOCUMENT_CONTEXTUALIZATION, PhaseStatus.COMPLETED, 1)

    def test_missing_conversion_fails_that_source_only(self):
        phase, deps = _phase()
        deps["repo"].get_by_source_id.return_value = None

        result = phase.execute([DocumentConversionJobResult(source_id=uuid.uuid4(), success=True)])

        assert result.status is PhaseStatus.FAILED
        deps["contextualizer"].execute_document.assert_not_called()

    def test_model_failure_fails_the_phase_without_running_jobs(self):
        phase, deps = _phase()
        deps["async"].resolve_preload.side_effect = ModelManagerError("out of VRAM")
        phase.preload_model()

        result = phase.execute([DocumentConversionJobResult(source_id=uuid.uuid4(), success=True)])

        assert result.status is PhaseStatus.FAILED
        deps["job_manager"].run_job.assert_not_called()

    def test_contextualization_error_still_unloads_the_model(self):
        phase, deps = _phase()
        deps["contextualizer"].execute_document.side_effect = RuntimeError("generation failed")
        phase.preload_model()

        result = phase.execute([DocumentConversionJobResult(source_id=uuid.uuid4(), success=True)])

        assert result.status is PhaseStatus.FAILED
        deps["async"].trigger_unload.assert_called_once()

    def test_discard_releases_an_unresolved_preload_once(self):
        phase, deps = _phase()
        phase.preload_model()

        phase.discard_preload()
        phase.discard_preload()

        deps["async"].discard_preload.assert_called_once_with(deps["async"].trigger_preload.return_value,
                                                              deps["loader"])
