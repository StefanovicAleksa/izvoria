# Path: integration_tests/features/deep_research_report/test_report_repository_integration.py

import uuid

import pytest
from sqlalchemy import text as sa_text

from app.core.database.data.engine import Base
from app.core.database.service.manager import db_manager
from app.features.deep_research_report.data.postgres_report_repo import PostgresReportRepository
from app.features.deep_research_report.domain.entities import (
    CorpusDigest, EvidenceEntry, IndexedSource, OutlineSection, ReportSection, ResearchPlan, SubQuestion,
)
from app.features.deep_research_report.domain.enums import EvidenceOrigin, ReportStatus, SectionStatus
from app.features.rag_storage.data.models import VectorChunkModel

pytestmark = pytest.mark.integration


def _plan(report_id: uuid.UUID) -> ResearchPlan:
    questions = [
        SubQuestion(text="Who sealed the bag?", section_key="s1", order_index=0),
        SubQuestion(text="Where was it stored?", section_key="s2", order_index=1),
    ]
    return ResearchPlan(
        report_id=report_id,
        topic="Chain of custody",
        scope_note="Covers handling and storage.",
        sections=[
            OutlineSection(key="s1", title="Handling", order_index=0, brief="How it was handled.",
                           sub_question_ids=[questions[0].id]),
            OutlineSection(key="s2", title="Storage", order_index=1, brief="Where it was kept.",
                           sub_question_ids=[questions[1].id]),
        ],
        sub_questions=questions,
    )


def _corpus() -> CorpusDigest:
    return CorpusDigest(
        indexed_source_count=3,
        indexed_chunk_count=90,
        sources=[IndexedSource(source_id=uuid.uuid4(), name="Deposition", chunk_count=30)],
        sample_digests=["[SPEAKER_00 @ 00:00:01]: sealed."],
    )


def _evidence(count: int = 3, chunk_ids=None):
    chunk_ids = chunk_ids or [uuid.uuid4() for _ in range(count)]
    return [
        EvidenceEntry(
            doc_number=index + 1,
            chunk_id=chunk_ids[index],
            source_id=uuid.uuid4(),
            source_name=f"Recording {index + 1}",
            start_sec=index * 60,
            end_sec=index * 60 + 30,
            speaker_labels=["SPEAKER_01"],
            digest=f"excerpt {index + 1}",
            origin=EvidenceOrigin.FAN_OUT if index else EvidenceOrigin.SWEEP,
            found_by="Who sealed the bag?" if index else None,
            rerank_score=0.5,
        )
        for index in range(count)
    ]


class TestReportRepositoryIntegration:
    """
    Real Postgres. Exercises the full lifecycle a run commits through, because those commits are the
    only reason a crashed or cancelled report is resumable.
    """

    @pytest.fixture(scope="module", autouse=True)
    def setup_schema(self):
        Base.metadata.create_all(bind=db_manager._engine_wrapper._engine)  # noqa
        yield

    @pytest.fixture
    def repository(self, db_manager):
        return PostgresReportRepository(db_manager)

    def test_full_lifecycle_create_approve_gather_write_finalize(self, repository):
        report_id = uuid.uuid4()
        plan = _plan(report_id)

        created = repository.create_report("Chain of custody", plan, _corpus())
        assert created.status == ReportStatus.AWAITING_APPROVAL
        assert len(created.plan.sections) == 2

        approved = repository.replace_plan(
            report_id, plan.model_copy(update={"revision": 2})
        )
        assert approved.status == ReportStatus.GATHERING
        assert approved.plan.revision == 2

        repository.save_evidence_bank(report_id, _evidence(3), swept_source_count=7, scrub_enabled=False)
        gathering = repository.get_report(report_id)
        assert gathering.status == ReportStatus.WRITING
        assert [e.doc_number for e in gathering.evidence] == [1, 2, 3]
        assert gathering.swept_source_count == 7

        repository.save_section_assignments(report_id, [
            OutlineSection(key="s1", title="Handling", order_index=0, evidence_doc_numbers=[1, 2]),
            OutlineSection(key="s2", title="Storage", order_index=1, evidence_doc_numbers=[3]),
        ])

        repository.save_section(report_id, ReportSection(
            key="s1", title="Handling", order_index=0, status=SectionStatus.WRITTEN,
            markdown_body="Sealed [Doc 1][Doc 2].", cited_doc_numbers=[1, 2],
        ))
        repository.save_section(report_id, ReportSection(
            key="s2", title="Storage", order_index=1, status=SectionStatus.SKIPPED_NO_EVIDENCE,
        ))

        final = repository.finalize(report_id, "Summary [Doc 1].", "/reports/r.md")
        assert final.status == ReportStatus.COMPLETED
        assert final.markdown_path == "/reports/r.md"
        assert [s.key for s in final.written_sections] == ["s1"]
        assert [s.key for s in final.uncovered_sections] == ["s2"]
        assert final.cited_doc_numbers == [1, 2]
        assert final.completed_at is not None

    def test_the_corpus_digest_survives_a_round_trip(self, repository):
        report_id = uuid.uuid4()
        repository.create_report("t", _plan(report_id), _corpus())

        loaded = repository.get_report(report_id)

        assert loaded.corpus is not None
        assert loaded.corpus.indexed_chunk_count == 90
        assert loaded.corpus.sources[0].name == "Deposition"

    def test_resume_reads_back_the_bank_and_the_pending_sections(self, repository):
        """The commit that makes a resume cost two sections instead of a whole re-gather."""
        report_id = uuid.uuid4()
        repository.create_report("t", _plan(report_id), _corpus())
        repository.replace_plan(report_id, _plan(report_id).model_copy(update={"revision": 2}))
        repository.save_evidence_bank(report_id, _evidence(3), swept_source_count=3, scrub_enabled=False)
        repository.save_section(report_id, ReportSection(
            key="s1", title="Handling", order_index=0, status=SectionStatus.WRITTEN,
            markdown_body="Sealed [Doc 1].", cited_doc_numbers=[1],
        ))
        repository.set_status(report_id, ReportStatus.CANCELLED, "Cancelled by the user.")

        reopened = repository.get_report(report_id)

        assert reopened.status == ReportStatus.CANCELLED
        assert len(reopened.evidence) == 3
        pending = [s.key for s in reopened.sections if s.status == SectionStatus.PENDING]
        assert pending == ["s2"]

    def test_replacing_the_plan_removes_sections_the_user_deleted(self, repository):
        report_id = uuid.uuid4()
        repository.create_report("t", _plan(report_id), _corpus())
        trimmed = _plan(report_id)
        trimmed = trimmed.model_copy(update={
            "sections": trimmed.sections[:1],
            "sub_questions": trimmed.sub_questions[:1],
            "revision": 2,
        })

        repository.replace_plan(report_id, trimmed)

        reloaded = repository.get_report(report_id)
        assert [s.key for s in reloaded.sections] == ["s1"]
        assert len(reloaded.plan.sub_questions) == 1

    def test_deleting_a_report_cascades_to_its_sections_and_evidence(self, repository):
        report_id = uuid.uuid4()
        repository.create_report("t", _plan(report_id), _corpus())
        repository.save_evidence_bank(report_id, _evidence(2), swept_source_count=2, scrub_enabled=False)

        with db_manager.session() as session:
            session.execute(
                sa_text("DELETE FROM research_reports WHERE id = :rid"), {"rid": str(report_id)}
            )

        with db_manager.session() as session:
            sections = session.execute(
                sa_text("SELECT COUNT(*) FROM research_report_sections WHERE report_id = :rid"),
                {"rid": str(report_id)},
            ).scalar()
            evidence = session.execute(
                sa_text("SELECT COUNT(*) FROM research_report_evidence WHERE report_id = :rid"),
                {"rid": str(report_id)},
            ).scalar()

        assert sections == 0
        assert evidence == 0

    def test_evidence_survives_the_chunk_it_points_at_being_re_ingested(self, repository):
        """
        `chunk_id` is deliberately NOT a foreign key. Chunks cascade from `sources`, so re-ingesting a
        file deletes them -- and a signed-off report has to survive that, rendering its citations from
        the denormalised provenance instead of losing them.
        """
        report_id = uuid.uuid4()
        chunk_id = uuid.uuid4()
        source_id = uuid.uuid4()

        with db_manager.session() as session:
            session.execute(
                sa_text("ALTER TABLE vector_chunks DROP CONSTRAINT IF EXISTS vector_chunks_source_id_fkey;")
            )
            session.add(VectorChunkModel(
                source_id=source_id, chunk_id=chunk_id, start_sec=0.0, end_sec=5.0,
                text_payload="context\n\n[SPEAKER_00 @ 00:00:00]: sealed.",
                raw_text="[SPEAKER_00 @ 00:00:00]: sealed.",
                context_text="context", speaker_labels=["SPEAKER_00"], embedding=[0.01] * 1024,
            ))

        repository.create_report("t", _plan(report_id), _corpus())
        repository.save_evidence_bank(
            report_id, _evidence(1, chunk_ids=[chunk_id]), swept_source_count=1, scrub_enabled=False
        )

        # The re-ingest.
        with db_manager.session() as session:
            session.execute(
                sa_text("DELETE FROM vector_chunks WHERE chunk_id = :cid"), {"cid": str(chunk_id)}
            )

        reopened = repository.get_report(report_id)

        assert len(reopened.evidence) == 1
        entry = reopened.evidence[0]
        assert entry.chunk_id == chunk_id
        assert entry.source_name == "Recording 1"
        assert entry.digest == "excerpt 1"
        assert entry.start_sec == 0

    def test_list_reports_is_newest_first_and_scopeable_to_a_session(self, repository):
        session_id = uuid.uuid4()
        first, second = uuid.uuid4(), uuid.uuid4()
        repository.create_report("first", _plan(first), _corpus(), session_id=session_id)
        repository.create_report("second", _plan(second), _corpus(), session_id=session_id)
        repository.create_report("unrelated", _plan(uuid.uuid4()), _corpus())

        scoped = repository.list_reports(session_id=session_id)

        assert {r.topic for r in scoped} == {"first", "second"}
        assert all(r.section_count == 2 for r in scoped)

    def test_a_missing_report_reads_back_as_none(self, repository):
        assert repository.get_report(uuid.uuid4()) is None

    def test_each_user_sees_only_their_own_reports(self, repository):
        """The hosted API: users of one firm share a database, but a report is its scoper's."""
        mine, theirs, shared = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        repository.create_report("mine", _plan(mine), _corpus())
        repository.create_report("theirs", _plan(theirs), _corpus())
        repository.create_report("appliance", _plan(shared), _corpus())
        repository.set_owner(mine, "alice")
        repository.set_owner(theirs, "bob")

        assert repository.get_report(mine, owner_id="alice").topic == "mine"
        assert repository.get_report(theirs, owner_id="alice") is None
        assert repository.get_report(shared, owner_id="alice") is None
        listed = {r.id for r in repository.list_reports(owner_id="alice")}
        assert mine in listed and not listed & {theirs, shared}
        # The appliance (no owner) still sees everything
        assert {mine, theirs, shared} <= {r.id for r in repository.list_reports(limit=500)}
        assert repository.get_report(theirs).topic == "theirs"

    def test_setting_the_owner_of_a_missing_report_raises(self, repository):
        from app.features.deep_research_report.exceptions.database import ReportNotFoundError

        with pytest.raises(ReportNotFoundError):
            repository.set_owner(uuid.uuid4(), "alice")
