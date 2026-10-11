# Path: frontend/tests/ui/test_report_view_unit.py

import uuid

from dataclasses import replace

import pytest

pytest.importorskip("PySide6")

from izvoria_desktop.domain.entities import (  # noqa: E402
    Evidence, ProgressTick, ReportDocument, ReportOutlineDraft, ReportPlanDraft,
)
from izvoria_desktop.domain.enums import (  # noqa: E402
    JobProgressState, MediaKind, ReportMode, ReportRunStatus,
)
from izvoria_desktop.ui.views.report_view import ReportView  # noqa: E402
from izvoria_desktop.ui.widgets.report_document import ReportDocumentView  # noqa: E402

pytestmark = pytest.mark.gui

REPORT_ID = uuid.UUID("44444444-4444-4444-4444-444444444444")


def _evidence(doc_number: int = 1, clip_path=None) -> Evidence:
    return Evidence(
        doc_number=doc_number,
        chunk_id=uuid.uuid4(),
        source_id=uuid.uuid4(),
        start_sec=10.0,
        end_sec=40.0,
        excerpt="[speaker_0 @ 00:00:10]: The bag was sealed.",
        speakers=("speaker_0",),
        clip_path=clip_path,
        media_kind=MediaKind.CLIP if clip_path else MediaKind.UNRESOLVED,
    )


def _plan(sections: int = 2) -> ReportPlanDraft:
    return ReportPlanDraft(
        report_id=REPORT_ID,
        topic="chain of custody",
        scope_note="Covers handling and storage.",
        sections=tuple(
            ReportOutlineDraft(key=f"s{i + 1}", title=f"Section {i + 1}", brief="brief",
                               questions=(f"question {i + 1}",))
            for i in range(sections)
        ),
        indexed_source_count=3,
        indexed_chunk_count=90,
        expected_job_count=6,
        estimated_minutes=12,
    )


def _document(status=ReportRunStatus.COMPLETED, evidence=None) -> ReportDocument:
    return ReportDocument(
        report_id=REPORT_ID,
        topic="chain of custody",
        markdown="# chain of custody\n\nSealed [Doc 1](izvoria://doc/1).\n",
        status=status,
        mode=ReportMode.EXHAUSTIVE,
        evidence=evidence if evidence is not None else (_evidence(1),),
        markdown_path="/reports/r.md",
        section_count=2,
        uncovered_count=1,
        rounds_used=2,
    )


@pytest.fixture
def view(qt_app):
    widget = ReportView()
    widget.show()
    yield widget
    widget.close()
    widget.deleteLater()


class TestPlanApproval:
    def test_a_topic_requests_a_plan_with_the_chosen_mode(self, view):
        requested = []
        view.planRequested.connect(lambda topic, mode: requested.append((topic, mode)))
        view._topic.setText("chain of custody")
        view._mode.setCurrentIndex(1)  # Exhaustive

        view._on_plan()

        assert requested == [("chain of custody", ReportMode.EXHAUSTIVE)]

    def test_an_empty_topic_requests_nothing(self, view):
        requested = []
        view.planRequested.connect(lambda *args: requested.append(args))

        view._on_plan()

        assert requested == []

    def test_approving_the_plan_emits_the_edited_sections(self, view):
        started = []
        view.reportRequested.connect(lambda plan, mode: started.append((plan, mode)))
        view.show_plan(_plan())

        view._outline._cards[0]._title.setText("Renamed section")
        view._outline._on_start()

        plan, mode = started[0]
        assert plan.sections[0].title == "Renamed section"
        # The key must survive a rename: the engine re-attaches evidence by key across revisions.
        assert plan.sections[0].key == "s1"
        assert mode in (ReportMode.DEEP, ReportMode.EXHAUSTIVE)

    def test_the_proposed_title_can_be_edited_before_approval(self, view):
        started = []
        view.reportRequested.connect(lambda plan, mode: started.append(plan))
        view.show_plan(replace(_plan(), title="Chain of Custody"))

        assert view._outline._title.text() == "Chain of Custody"
        view._outline._title.setText("  Evidence   Handling ")
        view._outline._on_start()

        assert started[0].title == "Evidence Handling"
        assert started[0].topic == "chain of custody"

    def test_history_lists_reports_by_title_and_untitled_ones_by_request(self, view):
        from izvoria_desktop.domain.entities import ReportRef
        shown = []
        view._history.set_entries = lambda entries: shown.extend(e.title for e in entries)

        view.set_reports([
            ReportRef(report_id=REPORT_ID, topic="create a report on custody", status=ReportRunStatus.COMPLETED,
                      title="Chain of Custody"),
            ReportRef(report_id=uuid.uuid4(), topic="an old request", status=ReportRunStatus.COMPLETED),
        ])

        assert shown == ["Chain of Custody", "an old request"]

    def test_dropping_a_section_removes_it_from_the_approved_plan(self, view):
        started = []
        view.reportRequested.connect(lambda plan, mode: started.append(plan))
        view.show_plan(_plan(sections=3))

        view._outline._on_remove("s2")
        view._outline._on_start()

        assert [s.key for s in started[0].sections] == ["s1", "s3"]

    def test_the_last_section_cannot_be_dropped(self, view):
        view.show_plan(_plan(sections=1))

        view._outline._on_remove("s1")

        assert len(view._outline._cards) == 1
        assert view._outline._warning.isVisible()

    def test_an_untitled_section_blocks_the_run(self, view):
        started = []
        view.reportRequested.connect(lambda *args: started.append(args))
        view.show_plan(_plan())
        view._outline._cards[0]._title.setText("   ")

        view._outline._on_start()

        assert started == []
        assert view._outline._warning.isVisible()


class TestPlanTransition:
    """
    The approval step has to be visibly distinct. Scoping takes about a minute; if the previous
    outline stays on screen the whole time and the new one swaps in underneath it, nothing appears
    to happen -- for the same topic the two outlines look almost identical, and the run reads as
    though it started itself.
    """

    def test_scoping_clears_the_previous_outline(self, view):
        view.show_plan(_plan())
        assert view._outline.isVisibleTo(view)

        view.start_planning("a new topic")

        assert not view._outline.isVisibleTo(view)
        assert view._scoping.isVisibleTo(view)
        assert "a new topic" in view._scoping.text()

    def test_the_new_plan_replaces_the_scoping_notice(self, view):
        view.start_planning("a new topic")

        view.show_plan(_plan())

        assert not view._scoping.isVisibleTo(view)
        assert view._outline.isVisibleTo(view)

    def test_scoping_drops_the_stale_plan_so_stop_cannot_target_it(self, view):
        """The old report_id must not linger as a cancel target for a run that is not happening."""
        view.show_plan(_plan())

        view.start_planning("a new topic")

        assert view._plan is None

    def test_a_failed_scope_clears_the_notice(self, view):
        view.start_planning("a new topic")

        view.show_failure("The appliance is busy.")

        assert not view._scoping.isVisibleTo(view)
        assert view._failure.isVisibleTo(view)


class TestRunProgress:
    def test_the_bar_is_determinate_from_the_plans_job_count(self, view):
        """Unlike ingestion, a report knows up front how many steps it will take."""
        view.start_run(_plan())

        assert view._progress.maximum() == 6
        assert view._progress.value() == 0

    def test_completed_report_ticks_advance_the_bar(self, view):
        view.start_run(_plan())

        view.apply_tick(ProgressTick(phase="research_section", state=JobProgressState.COMPLETED,
                                     detail="Writing section 1/2: Handling"))

        assert view._progress.value() == 1
        assert "Writing section 1 of 2: Handling" == view._phase.text()

    def test_ticks_from_other_work_are_ignored(self, view):
        """
        One listener sees every job in the process. Without this filter an ingestion's ticks would
        drive the report's progress bar.
        """
        view.start_run(_plan())

        view.apply_tick(ProgressTick(phase="transcription", state=JobProgressState.COMPLETED))

        assert view._progress.value() == 0

    def test_ticks_are_ignored_when_no_run_is_showing(self, view):
        # An untouched QProgressBar reports -1, so the assertion is "unchanged", not "zero".
        before = view._progress.value()

        view.apply_tick(ProgressTick(phase="research_section", state=JobProgressState.COMPLETED))

        assert view._progress.value() == before
        assert view._steps_done == 0

    def test_stopping_asks_for_cancellation_and_says_it_is_not_instant(self, view):
        cancelled = []
        view.cancelRequested.connect(cancelled.append)
        view.show_plan(_plan())
        view.start_run(_plan())

        view._on_stop()

        assert cancelled == [REPORT_ID]
        assert view._stop.isEnabled() is False
        assert "section" in view._stop.text().lower()


class TestReportRendering:
    def test_a_finished_report_is_shown_with_its_counts(self, view):
        view.show_report(_document())

        assert "2 sections" in view._summary.text()
        assert "gathering round" not in view._summary.text()
        assert view._partial.isVisible() is False

    def test_a_cancelled_report_is_flagged_as_partial(self, view):
        """A stopped report is worth reading, but must not read as though it were whole."""
        view.show_report(_document(status=ReportRunStatus.CANCELLED))

        assert view._partial.isVisible()
        assert "stopped" in view._partial.text().lower()

    def test_the_first_citation_is_opened_automatically(self, view):
        """The point is to verify, not to hunt."""
        view.show_report(_document())

        assert view._source_panel._evidence is not None

    def test_clicking_a_citation_shows_it_in_the_source_panel(self, view):
        report = _document(evidence=(_evidence(1), _evidence(3)))
        view.show_report(report)

        view._on_citation(report.evidence[1])

        assert view._source_panel._evidence.doc_number == 3

    def test_a_citation_with_no_evidence_says_so_rather_than_doing_nothing(self, view):
        view.show_report(_document())

        view._on_citation_missing(9)

        assert view._failure.isVisible()
        assert "[Doc 9]" in view._failure.text()


class TestReportDocumentView:
    @pytest.fixture
    def document(self, qt_app):
        widget = ReportDocumentView()
        widget.show()
        yield widget
        widget.close()
        widget.deleteLater()

    def test_an_izvoria_doc_link_resolves_to_its_evidence(self, document):
        from PySide6.QtCore import QUrl

        clicked = []
        document.citationClicked.connect(clicked.append)
        document.show_report(_document(evidence=(_evidence(1), _evidence(7))))

        document._on_anchor(QUrl("izvoria://doc/7"))

        assert [e.doc_number for e in clicked] == [7]

    def test_a_doc_number_the_bank_lost_is_reported(self, document):
        from PySide6.QtCore import QUrl

        missing = []
        document.citationMissing.connect(missing.append)
        document.show_report(_document())

        document._on_anchor(QUrl("izvoria://doc/42"))

        assert missing == [42]

    def test_a_non_citation_url_is_ignored(self, document):
        from PySide6.QtCore import QUrl

        clicked, missing = [], []
        document.citationClicked.connect(clicked.append)
        document.citationMissing.connect(missing.append)
        document.show_report(_document())

        document._on_anchor(QUrl("https://example.com"))

        assert clicked == [] and missing == []

    def test_qt_never_opens_links_itself(self, document):
        """izvoria:// means nothing outside this app; handing it to a browser would be a dead end."""
        assert document.openLinks() is False
        assert document.openExternalLinks() is False


class TestHistory:
    @staticmethod
    def _ref(status, evidence_count=3, report_id=REPORT_ID):
        from izvoria_desktop.domain.entities import ReportRef
        return ReportRef(report_id=report_id, topic="chain of custody", status=status,
                         section_count=4, evidence_count=evidence_count)

    @pytest.mark.parametrize("status, signal_name", [
        (ReportRunStatus.COMPLETED, "savedReportRequested"),
        (ReportRunStatus.CANCELLED, "savedReportRequested"),
        (ReportRunStatus.WRITING, "savedReportRequested"),
        (ReportRunStatus.AWAITING_APPROVAL, "savedPlanRequested"),
    ])
    def test_choosing_a_report_asks_for_what_its_state_needs(self, view, status, signal_name):
        view.set_reports([self._ref(status)])
        emitted = []
        getattr(view, signal_name).connect(emitted.append)
        view._on_history_chosen(REPORT_ID)
        assert emitted == [REPORT_ID]

    def test_a_report_stopped_before_any_evidence_prefills_its_topic_instead(self, view):
        view.set_reports([self._ref(ReportRunStatus.CANCELLED, evidence_count=0)])
        emitted = []
        view.savedReportRequested.connect(emitted.append)
        view._on_history_chosen(REPORT_ID)
        assert emitted == []
        assert view._topic.text() == "chain of custody"
        assert "nothing to resume" in view._failure.text()

    @pytest.mark.parametrize("status, evidence_count, visible", [
        (ReportRunStatus.CANCELLED, 3, True),
        (ReportRunStatus.WRITING, 3, True),     # the app went away mid-run
        (ReportRunStatus.CANCELLED, 0, False),
        (ReportRunStatus.COMPLETED, 3, False),
    ])
    def test_resume_is_offered_only_where_there_is_a_bank_to_resume_from(self, view, status, evidence_count,
                                                                         visible):
        view.set_reports([self._ref(status, evidence_count)])
        view.show_report(_document(status=status))
        assert view._resume.isVisible() is visible

    def test_stop_during_a_resumed_run_cancels_that_report(self, view):
        view.set_reports([self._ref(ReportRunStatus.CANCELLED)])
        view.show_report(_document(status=ReportRunStatus.CANCELLED))
        cancelled = []
        view.cancelRequested.connect(cancelled.append)

        view.start_resume(REPORT_ID)
        view._on_stop()

        assert cancelled == [REPORT_ID]

    def test_history_cannot_navigate_away_from_a_running_report(self, view):
        other = uuid.uuid4()
        view.set_reports([self._ref(ReportRunStatus.COMPLETED, report_id=other)])
        view.start_run(_plan())
        emitted = []
        view.savedReportRequested.connect(emitted.append)
        view._on_history_chosen(other)
        assert emitted == []
