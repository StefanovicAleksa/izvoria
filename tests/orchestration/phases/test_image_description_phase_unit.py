# Path: tests/orchestration/phases/test_image_description_phase_unit.py

import uuid
from unittest.mock import Mock

from app.core.jobs.domain.enums import JobType
from app.core.model_manager.exceptions.base import ModelManagerError
from app.orchestration.domain.entities import ContextualRetrievalJobResult, ImageDescriptionJobResult
from app.orchestration.domain.enums import PhaseStatus, PipelinePhase
from app.orchestration.service.phases.image_description_phase import ImageDescriptionPhase


def _phase():
    deps = {name: Mock() for name in ("job_manager", "image", "visual", "contextualizer", "async", "loader")}
    deps["job_manager"].run_job.side_effect = lambda request, func, **kwargs: func(**kwargs)
    description = deps["image"].execute.return_value
    description.ocr_checked = True
    description.dropped_items = ["Go live 2026"]
    description.ocr_incomplete = False
    description.ocr_withheld_lines = ["Analbeig aftea"]
    description.id = uuid.uuid4()
    deps["contextualizer"].execute_image.return_value.id = uuid.uuid4()
    phase = ImageDescriptionPhase(
        job_manager=deps["job_manager"], image_api=deps["image"], visual_api=deps["visual"],
        contextualizer_api=deps["contextualizer"], async_model_service=deps["async"], loader=deps["loader"])
    return phase, deps


class TestImageDescriptionPhase:
    def test_describes_every_image_with_the_vlm_held_for_the_batch(self):
        phase, deps = _phase()
        first, second = uuid.uuid4(), uuid.uuid4()
        phase.preload_model()

        result = phase.execute([first, second])

        deps["async"].trigger_preload.assert_called_once_with(manifest=deps["visual"].manifest, loader=deps["loader"])
        deps["async"].resolve_preload.assert_called_once_with(deps["async"].trigger_preload.return_value)
        model = deps["async"].resolve_preload.return_value
        assert [c.args for c in deps["image"].execute.call_args_list] == [(first, model), (second, model)]
        description = deps["image"].execute.return_value
        deps["contextualizer"].execute_image.assert_called_with(
            second, excerpt=deps["image"].render.return_value,
            context_string=deps["image"].render_context.return_value)
        deps["image"].render.assert_called_with(description)
        request = deps["job_manager"].run_job.call_args.args[0]
        assert (request.job_type, request.source_id) == (JobType.IMAGE_DESCRIPTION, second)
        deps["async"].trigger_unload.assert_called_once_with(loaded_model=model, loader=deps["loader"], flush=True)
        deps["image"].release.assert_called_once_with()

        assert (result.phase, result.status, result.successful_count) == (
            PipelinePhase.IMAGE_DESCRIPTION, PhaseStatus.COMPLETED, 2)
        item = result.items[0]
        assert isinstance(item, ImageDescriptionJobResult) and isinstance(item, ContextualRetrievalJobResult)
        assert item.contextual_transcript_id == deps["contextualizer"].execute_image.return_value.id
        assert (item.ocr_checked, item.text_items_dropped, item.ocr_incomplete, item.ocr_lines_withheld) == (
            True, 1, False, 1)

    def test_the_job_row_records_scalars_only(self):
        phase, deps = _phase()
        phase.preload_model()

        phase.execute([uuid.uuid4()])

        func = deps["job_manager"].run_job.call_args.args[1]
        outcome = func(source_id=uuid.uuid4(), vlm_model=Mock())
        assert set(outcome) == {"_contextual_transcript_id", "image_description_id", "ocr_checked",
                                "text_items_dropped", "ocr_incomplete", "ocr_lines_withheld"}

    def test_one_failed_image_does_not_stop_the_batch(self):
        phase, deps = _phase()
        good = deps["image"].execute.return_value
        deps["image"].execute.side_effect = [RuntimeError("decode failed"), good]
        phase.preload_model()

        result = phase.execute([uuid.uuid4(), uuid.uuid4()])

        assert result.status is PhaseStatus.PARTIAL_SUCCESS
        assert [item.success for item in result.items] == [False, True]
        assert "decode failed" in result.items[0].error_message
        deps["async"].trigger_unload.assert_called_once()

    def test_model_failure_fails_the_phase_without_running_jobs(self):
        phase, deps = _phase()
        deps["async"].resolve_preload.side_effect = ModelManagerError("out of VRAM")
        phase.preload_model()

        result = phase.execute([uuid.uuid4()])

        assert result.status is PhaseStatus.FAILED
        deps["job_manager"].run_job.assert_not_called()
        deps["async"].trigger_unload.assert_not_called()

    def test_without_a_preload_it_preloads_itself_once(self):
        phase, deps = _phase()

        phase.execute([uuid.uuid4()])

        deps["async"].trigger_preload.assert_called_once()
        deps["async"].trigger_unload.assert_called_once()

    def test_a_second_preload_discards_the_first(self):
        phase, deps = _phase()
        first = Mock()
        deps["async"].trigger_preload.side_effect = [first, Mock()]

        phase.preload_model()
        phase.preload_model()

        deps["async"].discard_preload.assert_called_once_with(first, deps["loader"])

    def test_discard_releases_an_unresolved_preload_once(self):
        phase, deps = _phase()
        phase.preload_model()

        phase.discard_preload()
        phase.discard_preload()

        deps["async"].discard_preload.assert_called_once_with(deps["async"].trigger_preload.return_value,
                                                              deps["loader"])
