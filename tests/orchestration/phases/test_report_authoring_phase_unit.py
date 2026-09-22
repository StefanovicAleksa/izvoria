# Path: tests/orchestration/phases/test_report_authoring_phase_unit.py

import uuid
from unittest.mock import MagicMock, Mock

import pytest

from app.core.jobs.domain.entities import CancellationToken
from app.core.jobs.domain.enums import JobType
from app.core.jobs.exceptions.execution import JobCancelled
from app.core.model_manager.domain.entities import LoadedModel, ModelManifest
from app.core.model_manager.domain.enums import ModelType
from app.core.model_manager.exceptions.base import ModelManagerError
from app.features.deep_research_report.domain.entities import (
    EvidenceEntry, FollowUpQuery, OutlineAssignment, OutlineSection, ReportSection, ResearchPlan,
)
from app.features.deep_research_report.domain.enums import SectionStatus
from app.orchestration.domain.entities import ReportSectionJobResult
from app.orchestration.domain.enums import PhaseStatus
from app.orchestration.service.phases.report_authoring_phase import ReportAuthoringPhase

REPORT_ID = uuid.uuid4()


def _plan() -> ResearchPlan:
    return ResearchPlan(
        report_id=REPORT_ID,
        topic="Chain of custody",
        sections=[
            OutlineSection(key="s1", title="Handling", order_index=0, evidence_doc_numbers=[1]),
            OutlineSection(key="s2", title="Storage", order_index=1, evidence_doc_numbers=[2]),
        ],
    )


def _evidence():
    return [
        EvidenceEntry(doc_number=n, chunk_id=uuid.uuid4(), source_id=uuid.uuid4(),
                      source_name="Deposition", start_sec=0, end_sec=30)
        for n in (1, 2)
    ]


def _section(key: str, status=SectionStatus.WRITTEN) -> ReportSection:
    return ReportSection(
        key=key, title=key.upper(), order_index=0, status=status,
        markdown_body="Body [Doc 1].", cited_doc_numbers=[1],
    )


@pytest.fixture
def job_manager():
    manager = Mock()
    manager.run_job.side_effect = lambda request, func, *args, cancel_token=None, **kwargs: func(
        *args, **kwargs
    )
    return manager


@pytest.fixture
def report_api():
    api = Mock()
    api.optimize_outline.return_value = OutlineAssignment(sections=_plan().sections)
    # The folded summary reads written sections back from the DB, not from this run's results, so a
    # resumed report's summary covers all of them.
    api.get_report.return_value = Mock(written_sections=[_section("s1"), _section("s2")])
    api.write_section.side_effect = lambda **kwargs: _section(kwargs["section"].key)
    api.write_summary.return_value = "Summary [Doc 1]."
    return api


@pytest.fixture
def async_service():
    service = Mock()
    service.resolve_preload.return_value = LoadedModel(
        manifest=ModelManifest(
            model_type=ModelType.REPORT_AUTHORING, model_path="models/x.gguf", required_vram_mb=1.0
        ),
        model_instance=MagicMock(),
    )
    return service


@pytest.fixture
def loader():
    return Mock()


@pytest.fixture
def phase(job_manager, report_api, async_service, loader):
    return ReportAuthoringPhase(job_manager, report_api, async_service, loader)


class TestModelLifecycle:
    def test_one_resolved_preload_covers_the_outline_pass_and_every_section(
            self, phase, async_service, report_api
    ):
        """
        The reason this is a single execute_report rather than two execute calls: two would mean two
        `finally` unloads and two loads of a ~5.7 GB file.
        """
        phase.preload_model()

        phase.execute_report(REPORT_ID, _plan(), _evidence(), {})

        assert async_service.resolve_preload.call_count == 1
        assert async_service.trigger_unload.call_count == 1
        assert report_api.optimize_outline.call_count == 1
        assert report_api.write_section.call_count == 2

    def test_release_after_is_false_while_the_phase_holds_the_model(self, phase, report_api):
        phase.preload_model()

        phase.execute_report(REPORT_ID, _plan(), _evidence(), {})

        assert report_api.optimize_outline.call_args.kwargs["release_after"] is False
        for call in report_api.write_section.call_args_list:
            assert call.kwargs["release_after"] is False

    def test_without_a_preload_the_api_releases_after_each_call(self, phase, report_api):
        phase.execute_report(REPORT_ID, _plan(), _evidence(), {})

        assert report_api.optimize_outline.call_args.kwargs["release_after"] is True

    def test_the_model_is_unloaded_even_when_a_section_raises(self, phase, async_service, report_api):
        phase.preload_model()
        report_api.write_section.side_effect = RuntimeError("llama_cpp exploded")

        phase.execute_report(REPORT_ID, _plan(), _evidence(), {})

        async_service.trigger_unload.assert_called_once()
        assert async_service.trigger_unload.call_args.kwargs["flush"] is True

    def test_a_failed_preload_fails_the_phase_without_writing_anything(
            self, phase, async_service, report_api
    ):
        phase.preload_model()
        async_service.resolve_preload.side_effect = ModelManagerError("no VRAM")

        result = phase.execute_report(REPORT_ID, _plan(), _evidence(), {})

        assert result.status == PhaseStatus.FAILED
        report_api.write_section.assert_not_called()

    def test_discard_preload_releases_an_unresolved_future(self, phase, async_service, loader):
        phase.preload_model()
        future = phase._preload_future

        phase.discard_preload()

        async_service.discard_preload.assert_called_once_with(future, loader)

    def test_discard_preload_is_a_no_op_without_one(self, phase, async_service):
        phase.discard_preload()

        async_service.discard_preload.assert_not_called()


class TestProgressReporting:
    def test_one_job_per_section_so_progress_ticks_even_though_the_model_never_moves(
            self, phase, job_manager
    ):
        phase.preload_model()

        phase.execute_report(REPORT_ID, _plan(), _evidence(), {})

        job_types = [c.args[0].job_type for c in job_manager.run_job.call_args_list]
        assert job_types == [
            JobType.RESEARCH_OUTLINE, JobType.RESEARCH_SECTION, JobType.RESEARCH_SECTION,
            # The summary rides the same residency now instead of reloading the 5.7 GB model.
            JobType.RESEARCH_SUMMARY,
        ]

    def test_each_section_job_carries_a_human_progress_label(self, phase, job_manager):
        phase.execute_report(REPORT_ID, _plan(), _evidence(), {})

        details = [c.args[0].progress_detail for c in job_manager.run_job.call_args_list]
        assert "Writing section 1/2: Handling" in details
        assert "Writing section 2/2: Storage" in details


class TestSectionOutcomes:
    def test_one_failing_section_does_not_lose_the_others(self, phase, report_api):
        def write(**kwargs):
            if kwargs["section"].key == "s1":
                raise RuntimeError("section blew up")
            return _section("s2")

        report_api.write_section.side_effect = write

        result = phase.execute_report(REPORT_ID, _plan(), _evidence(), {})

        assert result.status == PhaseStatus.PARTIAL_SUCCESS
        assert [s.key for s in phase.written_sections(result)] == ["s2"]

    def test_a_failed_outline_pass_falls_back_to_the_approved_outline(self, phase, report_api):
        """Losing a whole gathered bank because the assignment call failed would be a poor trade."""
        report_api.optimize_outline.side_effect = RuntimeError("bad json")

        result = phase.execute_report(REPORT_ID, _plan(), _evidence(), {})

        assert report_api.write_section.call_count == 2
        assert result.status == PhaseStatus.PARTIAL_SUCCESS

    def test_written_sections_helper_returns_only_written_ones(self, phase, report_api):
        report_api.write_section.side_effect = lambda **kwargs: _section(
            kwargs["section"].key,
            SectionStatus.WRITTEN if kwargs["section"].key == "s1" else SectionStatus.SKIPPED_NO_EVIDENCE,
        )

        result = phase.execute_report(REPORT_ID, _plan(), _evidence(), {})

        assert [s.key for s in phase.written_sections(result)] == ["s1"]

    def test_the_outline_context_tells_each_section_what_the_others_cover(self, phase, report_api):
        phase.execute_report(REPORT_ID, _plan(), _evidence(), {})

        context = report_api.write_section.call_args.kwargs["outline_context"]
        assert "1. Handling" in context
        assert "2. Storage" in context


class TestResume:
    def test_pending_keys_limit_the_run_to_unwritten_sections(self, phase, report_api):
        phase.execute_report(REPORT_ID, _plan(), _evidence(), {}, pending_keys=["s2"])

        assert report_api.write_section.call_count == 1
        assert report_api.write_section.call_args.kwargs["section"].key == "s2"

    def test_optimize_false_skips_the_outline_pass(self, phase, report_api):
        """A resume reuses the assignments already stored; re-running it would just cost a call."""
        phase.execute_report(REPORT_ID, _plan(), _evidence(), {}, optimize=False)

        report_api.optimize_outline.assert_not_called()
        assert report_api.write_section.call_count == 2


class TestCancellation:
    def test_a_cancelled_run_stops_between_sections_and_keeps_what_was_written(
            self, phase, report_api, job_manager
    ):
        token = CancellationToken()
        calls = {"n": 0}

        def run_job(request, func, *args, cancel_token=None, **kwargs):
            if request.job_type == JobType.RESEARCH_SECTION:
                calls["n"] += 1
                if calls["n"] == 2:
                    raise JobCancelled("stopped")
            return func(*args, **kwargs)

        job_manager.run_job.side_effect = run_job

        result = phase.execute_report(REPORT_ID, _plan(), _evidence(), {}, cancel_token=token)

        assert phase.was_cancelled(result) is True
        assert [s.key for s in phase.written_sections(result)] == ["s1"]

    def test_cancelling_during_the_outline_pass_propagates_and_still_unloads(
            self, phase, async_service, job_manager
    ):
        """
        Cancelled before any section exists, so there is nothing partial to keep: the phase lets
        JobCancelled out for the pipeline to record, and the `finally` still frees the VRAM. (Cancelled
        *between* sections is the other case above -- there the written sections are worth keeping.)
        """
        phase.preload_model()
        job_manager.run_job.side_effect = JobCancelled("stopped")

        with pytest.raises(JobCancelled):
            phase.execute_report(REPORT_ID, _plan(), _evidence(), {}, cancel_token=CancellationToken())

        async_service.trigger_unload.assert_called_once()
        assert async_service.trigger_unload.call_args.kwargs["flush"] is True

    def test_the_token_is_passed_to_every_gathering_and_writing_job(self, phase, job_manager):
        """The summary is exempt: it is skipped outright on cancellation rather than token-checked."""
        token = CancellationToken()

        phase.execute_report(REPORT_ID, _plan(), _evidence(), {}, cancel_token=token)

        cancellable = [
            c for c in job_manager.run_job.call_args_list
            if c.args[0].job_type is not JobType.RESEARCH_SUMMARY
        ]
        assert cancellable
        for call in cancellable:
            assert call.kwargs["cancel_token"] is token

    def test_was_cancelled_is_false_for_a_clean_run(self, phase):
        result = phase.execute_report(REPORT_ID, _plan(), _evidence(), {})

        assert phase.was_cancelled(result) is False


class TestSummary:
    def test_the_summary_rides_the_writing_residency_instead_of_reloading(self, phase, report_api):
        """
        It used to be a separate phase method that reloaded a 5.7 GB model for one short call -- a wasted
        heavy load in every report, in both modes.
        """
        phase.preload_model()

        result = phase.execute_report(REPORT_ID, _plan(), _evidence(), {})

        assert phase.summary(result) == "Summary [Doc 1]."
        assert report_api.write_summary.call_args.kwargs["release_after"] is False

    def test_the_summary_covers_every_written_section_not_just_this_runs(self, phase, report_api):
        """
        A resumed report writes two of eight sections. Summarising only those would silently produce a
        summary of a fragment.
        """
        phase.execute_report(REPORT_ID, _plan(), _evidence(), {}, pending_keys=["s2"])

        report_api.get_report.assert_called_with(REPORT_ID)
        assert [s.key for s in report_api.write_summary.call_args.kwargs["sections"]] == ["s1", "s2"]

    def test_summary_failure_does_not_lose_the_report(self, phase, report_api):
        """A report without a summary is still a report; losing the sections over it would not be."""
        report_api.write_summary.side_effect = RuntimeError("model died")

        result = phase.execute_report(REPORT_ID, _plan(), _evidence(), {})

        assert phase.summary(result) == ""
        assert len(phase.written_sections(result)) == 2
        assert result.status == PhaseStatus.COMPLETED

    def test_no_summary_is_written_when_the_run_was_cancelled(self, phase, report_api, job_manager):
        """The cancel path has to reach _finish_cancelled without waiting on a summary."""
        calls = {"n": 0}

        def run_job(request, func, *args, cancel_token=None, **kwargs):
            if request.job_type == JobType.RESEARCH_SECTION:
                calls["n"] += 1
                if calls["n"] == 2:
                    raise JobCancelled("stopped")
            return func(*args, **kwargs)

        job_manager.run_job.side_effect = run_job

        phase.execute_report(REPORT_ID, _plan(), _evidence(), {}, cancel_token=CancellationToken())

        report_api.write_summary.assert_not_called()

    def test_no_summary_when_nothing_was_ever_written(self, phase, report_api):
        report_api.get_report.return_value = Mock(written_sections=[])

        result = phase.execute_report(REPORT_ID, _plan(), _evidence(), {})

        assert phase.summary(result) == ""
        report_api.write_summary.assert_not_called()


class TestStopOnGaps:
    """The exhaustive mode's loop hinges on this: assign, then hand control back WITHOUT writing."""

    def _with_gap(self, report_api):
        report_api.optimize_outline.return_value = OutlineAssignment(
            sections=[
                OutlineSection(key="s1", title="Handling", order_index=0, evidence_doc_numbers=[1]),
                OutlineSection(key="s2", title="Storage", order_index=1),
            ],
            follow_ups=[FollowUpQuery(section_key="s2", text="re-worded query")],
        )

    def test_an_unsupported_section_stops_the_run_before_writing(self, phase, report_api):
        self._with_gap(report_api)

        result = phase.execute_report(REPORT_ID, _plan(), _evidence(), {}, stop_on_gaps=True)

        assert phase.stopped_for_gaps(result) is True
        report_api.write_section.assert_not_called()
        assert [f.text for f in phase.follow_ups(result)] == ["re-worded query"]

    def test_the_early_return_is_completed_not_failed(self, phase, report_api):
        """The pipeline raises WorkflowStateError on FAILED, which would abort the loop."""
        self._with_gap(report_api)

        result = phase.execute_report(REPORT_ID, _plan(), _evidence(), {}, stop_on_gaps=True)

        assert result.status == PhaseStatus.COMPLETED

    def test_the_model_is_still_released_when_stopping_for_gaps(self, phase, report_api, async_service):
        """
        The whole reason to stop here is to free VRAM so the caller can gather again. If the model stayed
        resident, the next acquire_model(wait_for_clear=True) would park forever.
        """
        self._with_gap(report_api)
        phase.preload_model()

        phase.execute_report(REPORT_ID, _plan(), _evidence(), {}, stop_on_gaps=True)

        async_service.trigger_unload.assert_called_once()
        assert async_service.trigger_unload.call_args.kwargs["flush"] is True

    def test_a_fully_covered_outline_writes_straight_through(self, phase, report_api):
        result = phase.execute_report(REPORT_ID, _plan(), _evidence(), {}, stop_on_gaps=True)

        assert phase.stopped_for_gaps(result) is False
        assert report_api.write_section.call_count == 2

    def test_a_failed_outline_pass_is_not_mistaken_for_gaps(self, phase, report_api):
        """
        A failed assignment falls back to the approved outline, whose evidence lists are empty by
        default. Reading that as "every section is a gap" would burn the whole round budget
        re-gathering evidence that was already fine.
        """
        report_api.optimize_outline.side_effect = RuntimeError("bad json")

        result = phase.execute_report(REPORT_ID, _plan(), _evidence(), {}, stop_on_gaps=True)

        assert phase.stopped_for_gaps(result) is False
        assert report_api.write_section.call_count == 2

    def test_stop_on_gaps_defaults_off_so_deep_mode_is_unchanged(self, phase, report_api):
        self._with_gap(report_api)

        result = phase.execute_report(REPORT_ID, _plan(), _evidence(), {})

        assert phase.stopped_for_gaps(result) is False
        assert report_api.write_section.call_count == 2
