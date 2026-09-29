# Path: tests/features/deep_research_report/data/test_postgres_report_repo.py

import uuid
from unittest.mock import MagicMock, Mock

import pytest

from app.core.database import exceptions as db_exc
from app.features.deep_research_report.data.postgres_report_repo import PostgresReportRepository
from app.features.deep_research_report.domain.entities import (
    CorpusDigest, EvidenceEntry, IndexedSource, OutlineSection, ReportSection, ResearchPlan, SubQuestion,
)
from app.features.deep_research_report.domain.enums import (
    EvidenceOrigin, ReportMode, ReportStatus, SectionStatus,
)
from app.features.deep_research_report.exceptions.database import ReportNotFoundError, ReportPersistenceError

REPORT_ID = uuid.UUID("33333333-3333-3333-3333-333333333333")


def _plan(report_id=REPORT_ID) -> ResearchPlan:
    return ResearchPlan(
        report_id=report_id,
        topic="Chain of custody",
        scope_note="Covers handling.",
        sections=[
            OutlineSection(key="s1", title="Handling", order_index=0, brief="How it was handled."),
            OutlineSection(key="s2", title="Storage", order_index=1),
        ],
        sub_questions=[SubQuestion(text="Who sealed it?", section_key="s1", order_index=0)],
        revision=2,
    )


def _corpus() -> CorpusDigest:
    return CorpusDigest(
        indexed_source_count=3,
        indexed_chunk_count=90,
        sources=[IndexedSource(source_id=uuid.uuid4(), name="Deposition", chunk_count=30)],
        sample_digests=["[SPEAKER_00 @ 00:00:01]: sealed."],
    )


def _report_row(report_id=REPORT_ID, **overrides) -> MagicMock:
    """
    A row whose scalar fields are real values.

    The write methods hydrate and return a Report before committing, so a bare MagicMock row fails
    pydantic validation on the way out rather than on the behaviour under test.
    """
    row = MagicMock()
    row.id = report_id
    row.session_id = None
    row.topic = "Chain of custody"
    row.status = ReportStatus.AWAITING_APPROVAL.value
    row.scope_note = "Covers handling."
    row.plan_revision = 1
    row.executive_summary = ""
    row.markdown_path = None
    row.corpus_digest = {}
    row.indexed_source_count = 3
    row.swept_source_count = 0
    row.scrub_enabled = False
    row.mode = ReportMode.DEEP.value
    row.rounds_used = 1
    row.language = "en"
    row.error_message = None
    row.created_at = None
    row.updated_at = None
    row.completed_at = None
    for key, value in overrides.items():
        setattr(row, key, value)
    return row


class TestPostgresReportRepository:
    @pytest.fixture
    def session(self):
        session = MagicMock()
        # _to_report() runs three child selects; empty is the right default for a fresh report.
        session.execute.return_value.scalars.return_value.all.return_value = []
        session.get.return_value = _report_row()
        return session

    @pytest.fixture
    def db_manager(self, session):
        manager = Mock()
        ctx = MagicMock()
        ctx.__enter__.return_value = session
        manager.session.return_value = ctx
        return manager

    @pytest.fixture
    def repository(self, db_manager):
        return PostgresReportRepository(db_manager)

    @staticmethod
    def _added(session, tablename: str):
        return [
            call.args[0] for call in session.add.call_args_list
            if getattr(type(call.args[0]), "__tablename__", None) == tablename
        ]

    # ==========================================
    # CREATE
    # ==========================================

    def test_create_report_persists_the_plan_and_the_corpus_digest(self, repository, session):
        repository.create_report("Chain of custody", _plan(), _corpus(), session_id=None)

        report = self._added(session, "research_reports")[0]
        # A real ResearchReportModel, so these are the values that would reach Postgres.
        assert report.id == REPORT_ID
        assert report.topic == "Chain of custody"
        assert report.status == ReportStatus.AWAITING_APPROVAL.value
        assert report.indexed_source_count == 3
        assert report.swept_source_count == 0
        # Stored, not recomputed: the report's coverage claim must stay true as the corpus grows.
        assert report.corpus_digest["indexed_chunk_count"] == 90

    def test_create_report_writes_a_row_per_section_and_sub_question(self, repository, session):
        repository.create_report("t", _plan(), _corpus())

        assert [s.section_key for s in self._added(session, "research_report_sections")] == ["s1", "s2"]
        questions = self._added(session, "research_report_sub_questions")
        assert [q.question_text for q in questions] == ["Who sealed it?"]
        assert questions[0].section_key == "s1"

    def test_create_report_links_a_session_when_one_is_given(self, repository, session):
        session_id = uuid.uuid4()

        repository.create_report("t", _plan(), _corpus(), session_id=session_id)

        assert self._added(session, "research_reports")[0].session_id == session_id

    def test_create_report_records_the_language_it_is_written_in(self, repository, session):
        created = repository.create_report("t", _plan(), _corpus(), language="sr")

        assert self._added(session, "research_reports")[0].language == "sr"
        assert created.language == "sr"

    # ==========================================
    # APPROVE
    # ==========================================

    def test_replace_plan_deletes_the_old_outline_wholesale(self, repository, session):
        """
        Replaced, not merged: a section the user deleted must not survive as a stale row that later
        gets evidence assigned to it.
        """
        repository.replace_plan(REPORT_ID, _plan())

        assert session.execute.call_count >= 2  # sections + sub-questions deleted
        assert [s.section_key for s in self._added(session, "research_report_sections")] == ["s1", "s2"]

    def test_replace_plan_moves_the_report_to_gathering_and_bumps_the_revision(self, repository, session):
        report = _report_row()
        session.get.return_value = report

        repository.replace_plan(REPORT_ID, _plan())

        assert report.status == ReportStatus.GATHERING.value
        assert report.plan_revision == 2

    def test_replace_plan_on_a_missing_report_raises(self, repository, session):
        session.get.return_value = None

        with pytest.raises(ReportNotFoundError):
            repository.replace_plan(REPORT_ID, _plan())

    # ==========================================
    # EVIDENCE BANK
    # ==========================================

    def test_save_evidence_bank_persists_every_entry_and_moves_to_writing(self, repository, session):
        report = _report_row()
        session.get.return_value = report
        evidence = [
            EvidenceEntry(doc_number=n, chunk_id=uuid.uuid4(), source_id=uuid.uuid4(),
                          source_name="Deposition", start_sec=n * 10, end_sec=n * 10 + 30,
                          speaker_labels=["SPEAKER_01"], digest="d", origin=EvidenceOrigin.SWEEP)
            for n in (1, 2, 3)
        ]

        repository.save_evidence_bank(REPORT_ID, evidence, swept_source_count=7, scrub_enabled=True)

        rows = self._added(session, "research_report_evidence")
        assert [r.doc_number for r in rows] == [1, 2, 3]
        assert rows[0].source_name == "Deposition"
        assert rows[0].origin == "sweep"
        assert report.swept_source_count == 7
        assert report.scrub_enabled is True
        assert report.mode == ReportMode.DEEP.value
        assert report.rounds_used == 1
        assert report.status == ReportStatus.WRITING.value

    def test_save_evidence_bank_replaces_any_previous_bank(self, repository, session):
        repository.save_evidence_bank(REPORT_ID, [], swept_source_count=0, scrub_enabled=False)

        session.execute.assert_called_once()  # the DELETE

    # ==========================================
    # SECTIONS
    # ==========================================

    def test_save_section_writes_the_body_and_its_citations(self, repository, session):
        row = MagicMock()
        session.execute.return_value.scalar_one_or_none.return_value = row
        section = ReportSection(
            key="s1", title="Handling", order_index=0, status=SectionStatus.WRITTEN,
            markdown_body="Sealed [Doc 1].", cited_doc_numbers=[1, 3],
        )

        repository.save_section(REPORT_ID, section)

        assert row.status == "written"
        assert row.markdown_body == "Sealed [Doc 1]."
        assert row.cited_doc_numbers == [1, 3]
        assert row.written_at is not None

    def test_save_section_for_an_unknown_key_raises(self, repository, session):
        session.execute.return_value.scalar_one_or_none.return_value = None
        section = ReportSection(key="ghost", title="G", order_index=0)

        with pytest.raises(ReportNotFoundError):
            repository.save_section(REPORT_ID, section)

    def test_save_section_assignments_updates_the_matching_rows(self, repository, session):
        row = MagicMock()
        row.section_key = "s1"
        session.execute.return_value.scalars.return_value.all.return_value = [row]

        repository.save_section_assignments(
            REPORT_ID, [OutlineSection(key="s1", title="H", order_index=3, evidence_doc_numbers=[2, 5])]
        )

        assert row.evidence_doc_numbers == [2, 5]
        assert row.order_index == 3

    def test_an_assignment_for_an_unknown_section_is_ignored_not_fatal(self, repository, session):
        session.execute.return_value.scalars.return_value.all.return_value = []

        repository.save_section_assignments(
            REPORT_ID, [OutlineSection(key="ghost", title="G", order_index=0)]
        )

    # ==========================================
    # FINALIZE / STATUS
    # ==========================================

    def test_finalize_stores_the_summary_and_path_and_clears_any_error(self, repository, session):
        report = _report_row()
        session.get.return_value = report

        repository.finalize(REPORT_ID, "Summary [Doc 1].", "/reports/r.md")

        assert report.executive_summary == "Summary [Doc 1]."
        assert report.markdown_path == "/reports/r.md"
        assert report.status == ReportStatus.COMPLETED.value
        assert report.completed_at is not None
        assert report.error_message is None

    def test_set_status_records_a_terminal_state_with_its_reason(self, repository, session):
        report = _report_row()
        session.get.return_value = report

        repository.set_status(REPORT_ID, ReportStatus.CANCELLED, "Cancelled by the user.")

        assert report.status == ReportStatus.CANCELLED.value
        assert report.error_message == "Cancelled by the user."

    # ==========================================
    # READS
    # ==========================================

    def test_get_report_returns_none_for_an_unknown_id(self, repository, session):
        session.get.return_value = None

        assert repository.get_report(uuid.uuid4()) is None

    def test_list_reports_returns_an_empty_list_when_there_are_none(self, repository, session):
        session.execute.return_value.scalars.return_value.all.return_value = []

        assert repository.list_reports() == []

    # ==========================================
    # ERROR MAPPING
    # ==========================================

    def test_database_errors_become_typed_persistence_errors(self, repository, session):
        session.add.side_effect = db_exc.SystemError("connection refused")

        with pytest.raises(ReportPersistenceError):
            repository.create_report("t", _plan(), _corpus())
