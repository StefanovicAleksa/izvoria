# Path: tests/orchestration/phases/test_research_scoping_phase_unit.py

import uuid
from unittest.mock import MagicMock, Mock

import pytest

from app.core.jobs.domain.enums import JobType
from app.core.model_manager.domain.entities import LoadedModel, ModelManifest
from app.core.model_manager.domain.enums import ModelType
from app.core.model_manager.exceptions.base import ModelManagerError
from app.features.deep_research_report.domain.entities import (
    CorpusDigest, OutlineSection, Report, ResearchPlan, SubQuestion,
)
from app.features.deep_research_report.domain.enums import ReportStatus
from app.orchestration.domain.entities import ResearchScopingJobResult
from app.orchestration.domain.enums import PhaseStatus, PipelinePhase
from app.orchestration.service.phases.research_scoping_phase import ResearchScopingPhase

TOPIC = "Chain of custody"


def _corpus() -> CorpusDigest:
    return CorpusDigest(indexed_source_count=3, indexed_chunk_count=120, sample_digests=["a real excerpt"])


def _report(report_id=None, sections=2, sub_questions=3) -> Report:
    report_id = report_id or uuid.uuid4()
    keys = [f"s{index}" for index in range(sections)]
    return Report(
        id=report_id,
        topic=TOPIC,
        status=ReportStatus.AWAITING_APPROVAL,
        plan=ResearchPlan(
            report_id=report_id,
            topic=TOPIC,
            sections=[
                OutlineSection(key=key, title=key.upper(), order_index=index)
                for index, key in enumerate(keys)
            ],
            sub_questions=[
                SubQuestion(text=f"q{index}", section_key=keys[index % len(keys)], order_index=index)
                for index in range(sub_questions)
            ],
        ),
    )


@pytest.fixture
def job_manager():
    manager = Mock()
    manager.run_job.side_effect = lambda request, func, *args, **kwargs: func(*args, **kwargs)
    return manager


@pytest.fixture
def report_api():
    api = Mock()
    api.scope_plan.side_effect = lambda **kwargs: _report()
    return api


@pytest.fixture
def loaded_model():
    return LoadedModel(
        manifest=ModelManifest(
            model_type=ModelType.REPORT_AUTHORING, model_path="models/x.gguf", required_vram_mb=1.0
        ),
        model_instance=MagicMock(),
    )


@pytest.fixture
def async_service(loaded_model):
    service = Mock()
    service.resolve_preload.return_value = loaded_model
    return service


@pytest.fixture
def loader():
    return Mock()


@pytest.fixture
def phase(job_manager, report_api, async_service, loader):
    return ResearchScopingPhase(job_manager, report_api, async_service, loader)


class TestModelLifecycle:
    def test_a_resolved_preload_is_unloaded_when_the_phase_returns(
            self, phase, async_service, loaded_model, loader
    ):
        phase.preload_model()

        phase.execute([(TOPIC, _corpus(), None)])

        assert async_service.resolve_preload.call_count == 1
        async_service.trigger_unload.assert_called_once_with(
            loaded_model=loaded_model, loader=loader, flush=True
        )

    def test_release_after_is_false_while_the_phase_holds_the_model(self, phase, report_api):
        """The phase owns the residency, so the feature API must not release it out from under it."""
        phase.preload_model()

        phase.execute([(TOPIC, _corpus(), None)])

        assert report_api.scope_plan.call_args.kwargs["release_after"] is False

    def test_without_a_preload_the_api_releases_after_the_call(self, phase, report_api, async_service):
        phase.execute([(TOPIC, _corpus(), None)])

        assert report_api.scope_plan.call_args.kwargs["release_after"] is True
        async_service.trigger_unload.assert_not_called()

    def test_the_model_is_unloaded_even_when_scoping_raises(
            self, phase, async_service, report_api, loaded_model, loader
    ):
        phase.preload_model()
        report_api.scope_plan.side_effect = RuntimeError("llama_cpp exploded")

        phase.execute([(TOPIC, _corpus(), None)])

        async_service.trigger_unload.assert_called_once_with(
            loaded_model=loaded_model, loader=loader, flush=True
        )

    def test_a_failed_preload_fails_the_phase_without_running_the_job(
            self, phase, async_service, report_api
    ):
        phase.preload_model()
        async_service.resolve_preload.side_effect = ModelManagerError("out of VRAM")

        result = phase.execute([(TOPIC, _corpus(), None)])

        assert result.status == PhaseStatus.FAILED
        assert result.phase == PipelinePhase.RESEARCH_SCOPING
        report_api.scope_plan.assert_not_called()
        async_service.trigger_unload.assert_not_called()

    def test_preload_model_discards_an_unresolved_preload_before_firing_a_new_one(
            self, phase, async_service, loader
    ):
        """
        An unresolved preload that gets overwritten becomes unreachable: the worker still loads that
        model, nothing can release it, and the next wait_for_clear acquisition parks forever.
        """
        first = Mock()
        second = Mock()
        async_service.trigger_preload.side_effect = [first, second]

        phase.preload_model()
        phase.preload_model()

        async_service.discard_preload.assert_called_once_with(first, loader)

    def test_discard_preload_is_a_no_op_once_the_preload_has_been_resolved(self, phase, async_service):
        phase.preload_model()
        phase.execute([(TOPIC, _corpus(), None)])

        phase.discard_preload()

        async_service.discard_preload.assert_not_called()

    def test_discard_preload_releases_a_preload_execute_never_resolved(
            self, phase, async_service, loader
    ):
        future = Mock()
        async_service.trigger_preload.return_value = future
        phase.preload_model()

        phase.discard_preload()

        async_service.discard_preload.assert_called_once_with(future, loader)


class TestJobWrapping:
    def test_the_scoping_call_runs_as_a_research_scoping_job(self, phase, job_manager):
        phase.execute([(TOPIC, _corpus(), None)])

        request = job_manager.run_job.call_args.args[0]
        assert request.job_type == JobType.RESEARCH_SCOPING
        assert request.source_id is None
        assert request.input_params == {"topic": TOPIC}
        assert request.progress_detail == f"Planning the report: {TOPIC}"

    def test_the_topic_corpus_and_session_are_passed_through_to_the_api(self, phase, report_api):
        session_id = uuid.uuid4()
        corpus = _corpus()

        phase.execute([(TOPIC, corpus, session_id)])

        kwargs = report_api.scope_plan.call_args.kwargs
        assert kwargs["topic"] == TOPIC
        assert kwargs["corpus"] is corpus
        assert kwargs["session_id"] == session_id


class TestResults:
    def test_a_successful_scope_reports_its_section_and_sub_question_counts(self, phase, report_api):
        report = _report(sections=4, sub_questions=9)
        report_api.scope_plan.side_effect = lambda **kwargs: report

        result = phase.execute([(TOPIC, _corpus(), None)])

        assert result.status == PhaseStatus.COMPLETED
        assert result.successful_count == 1
        assert result.failed_count == 0
        job_result, payload = result.items[0]
        assert job_result == ResearchScopingJobResult(
            report_id=report.id, success=True, section_count=4, sub_question_count=9
        )
        assert payload is report

    def test_a_failure_is_recorded_as_a_failed_job_with_its_message(self, phase, report_api):
        report_api.scope_plan.side_effect = RuntimeError("the model returned no JSON")

        result = phase.execute([(TOPIC, _corpus(), None)])

        assert result.status == PhaseStatus.FAILED
        assert result.failed_count == 1
        job_result, payload = result.items[0]
        assert job_result.success is False
        assert job_result.report_id is None
        assert job_result.error_message == "the model returned no JSON"
        assert payload is None
        assert result.errors == [f"Failed scoping for topic '{TOPIC}': the model returned no JSON"]

    def test_one_failure_among_several_topics_is_partial_success(self, phase, report_api):
        report_api.scope_plan.side_effect = [_report(), RuntimeError("boom"), _report()]

        result = phase.execute([
            (TOPIC, _corpus(), None), ("second", _corpus(), None), ("third", _corpus(), None),
        ])

        assert result.status == PhaseStatus.PARTIAL_SUCCESS
        assert result.successful_count == 2
        assert result.failed_count == 1
        assert result.total_processed == 3

    def test_one_residency_covers_every_topic_in_the_batch(self, phase, async_service, report_api):
        """The batch shape exists so a multi-topic call costs one 5.7 GB load, not one per topic."""
        phase.preload_model()

        phase.execute([(TOPIC, _corpus(), None), ("second", _corpus(), None)])

        assert async_service.resolve_preload.call_count == 1
        assert async_service.trigger_unload.call_count == 1
        assert report_api.scope_plan.call_count == 2

    def test_an_empty_batch_completes_without_touching_the_api(self, phase, report_api):
        result = phase.execute([])

        assert result.status == PhaseStatus.COMPLETED
        assert result.items == []
        report_api.scope_plan.assert_not_called()
