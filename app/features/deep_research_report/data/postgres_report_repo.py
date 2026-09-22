# Path: app/features/deep_research_report/data/postgres_report_repo.py

import logging
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import delete, func, select

from app.core.database import exceptions as db_exc
from app.core.database.domain.interfaces import IDatabaseManager
from app.features.deep_research_report.data.models import (
    ResearchReportEvidenceModel, ResearchReportModel, ResearchReportSectionModel, ResearchReportSubQuestionModel,
)
from app.features.deep_research_report.domain.entities import (
    CorpusDigest, EvidenceEntry, OutlineSection, Report, ReportSection, ReportSummary, ResearchPlan, SubQuestion,
)
from app.features.deep_research_report.domain.enums import (
    EvidenceOrigin, ReportMode, ReportStatus, SectionStatus,
)
from app.features.deep_research_report.domain.interfaces import IReportRepository
from app.features.deep_research_report.exceptions.database import ReportNotFoundError
from app.features.deep_research_report.utils.error_mapper import map_report_error

logger = logging.getLogger(__name__)


class PostgresReportRepository(IReportRepository):
    """
    Persistence for reports. Each public method is one committed stage boundary of a run, which is what
    makes a crashed or cancelled report resumable rather than lost.
    """

    def __init__(self, db_manager: IDatabaseManager):
        self._db = db_manager

    # ==========================================
    # WRITES
    # ==========================================

    def create_report(
            self,
            topic: str,
            plan: ResearchPlan,
            corpus: CorpusDigest,
            session_id: Optional[uuid.UUID] = None,
    ) -> Report:
        try:
            with self._db.session() as session:
                report = ResearchReportModel(
                    id=plan.report_id,
                    session_id=session_id,
                    topic=topic,
                    status=ReportStatus.AWAITING_APPROVAL.value,
                    scope_note=plan.scope_note,
                    plan_revision=plan.revision,
                    corpus_digest=corpus.model_dump(mode="json"),
                    indexed_source_count=corpus.indexed_source_count,
                    swept_source_count=0,
                )
                session.add(report)
                self._write_plan_rows(session, plan)
                session.flush()
                return self._to_report(session, report)

        except db_exc.DatabaseError as e:
            raise map_report_error(e, context=f"Creating report for topic '{topic}'") from e

    def replace_plan(self, report_id: uuid.UUID, plan: ResearchPlan) -> Report:
        try:
            with self._db.session() as session:
                report = self._require(session, report_id)

                # Replaced wholesale, not merged: the approved outline is the plan of record, and a section
                # the user deleted must not survive as a stale row that later gets evidence assigned to it.
                session.execute(delete(ResearchReportSectionModel)
                                .where(ResearchReportSectionModel.report_id == report_id))
                session.execute(delete(ResearchReportSubQuestionModel)
                                .where(ResearchReportSubQuestionModel.report_id == report_id))

                self._write_plan_rows(session, plan)
                report.plan_revision = plan.revision
                report.scope_note = plan.scope_note
                report.status = ReportStatus.GATHERING.value
                session.flush()
                return self._to_report(session, report)

        except db_exc.DatabaseError as e:
            raise map_report_error(e, context=f"Replacing plan for report {report_id}") from e

    def save_evidence_bank(
            self,
            report_id: uuid.UUID,
            evidence: List[EvidenceEntry],
            swept_source_count: int,
            scrub_enabled: bool,
            mode: ReportMode = ReportMode.DEEP,
            rounds_used: int = 1,
    ) -> None:
        """
        The commit that ends gathering. After this, a resume needs no retrieval model at all -- the bank is
        the run's memory, and it is complete.
        """
        try:
            with self._db.session() as session:
                report = self._require(session, report_id)

                session.execute(delete(ResearchReportEvidenceModel)
                                .where(ResearchReportEvidenceModel.report_id == report_id))
                for entry in evidence:
                    session.add(ResearchReportEvidenceModel(
                        report_id=report_id,
                        doc_number=entry.doc_number,
                        chunk_id=entry.chunk_id,
                        source_id=entry.source_id,
                        source_name=entry.source_name,
                        start_sec=entry.start_sec,
                        end_sec=entry.end_sec,
                        speaker_labels=list(entry.speaker_labels),
                        digest=entry.digest,
                        origin=entry.origin.value,
                        found_by=entry.found_by,
                        rerank_score=entry.rerank_score,
                        scrubbed=entry.scrubbed,
                    ))

                report.swept_source_count = swept_source_count
                report.scrub_enabled = scrub_enabled
                report.mode = mode.value
                report.rounds_used = rounds_used
                report.status = ReportStatus.WRITING.value

        except db_exc.DatabaseError as e:
            raise map_report_error(e, context=f"Saving evidence bank for report {report_id}") from e

    def save_section_assignments(self, report_id: uuid.UUID, sections: List[OutlineSection]) -> None:
        try:
            with self._db.session() as session:
                by_key = {
                    row.section_key: row
                    for row in session.execute(
                        select(ResearchReportSectionModel)
                        .where(ResearchReportSectionModel.report_id == report_id)
                    ).scalars().all()
                }
                for section in sections:
                    row = by_key.get(section.key)
                    if row is None:
                        logger.warning(
                            f"Outline optimisation returned section {section.key!r}, which is not in report "
                            f"{report_id}; ignoring it."
                        )
                        continue
                    row.evidence_doc_numbers = list(section.evidence_doc_numbers)
                    row.order_index = section.order_index

        except db_exc.DatabaseError as e:
            raise map_report_error(e, context=f"Saving section assignments for report {report_id}") from e

    def save_section(self, report_id: uuid.UUID, section: ReportSection) -> None:
        """Committed the moment a section is finished, so a crash costs at most the section in flight."""
        try:
            with self._db.session() as session:
                row = session.execute(
                    select(ResearchReportSectionModel)
                    .where(ResearchReportSectionModel.report_id == report_id)
                    .where(ResearchReportSectionModel.section_key == section.key)
                ).scalar_one_or_none()

                if row is None:
                    raise ReportNotFoundError(
                        f"Section {section.key!r} does not exist on report {report_id}."
                    )

                row.status = section.status.value
                row.markdown_body = section.markdown_body
                row.cited_doc_numbers = list(section.cited_doc_numbers)
                row.written_at = section.written_at or datetime.now(timezone.utc)

        except db_exc.DatabaseError as e:
            raise map_report_error(e, context=f"Saving section {section.key} of report {report_id}") from e

    def finalize(self, report_id: uuid.UUID, executive_summary: str, markdown_path: str) -> Report:
        try:
            with self._db.session() as session:
                report = self._require(session, report_id)
                report.executive_summary = executive_summary
                report.markdown_path = markdown_path
                report.status = ReportStatus.COMPLETED.value
                report.completed_at = datetime.now(timezone.utc)
                report.error_message = None
                session.flush()
                return self._to_report(session, report)

        except db_exc.DatabaseError as e:
            raise map_report_error(e, context=f"Finalizing report {report_id}") from e

    def set_status(
            self,
            report_id: uuid.UUID,
            status: ReportStatus,
            error_message: Optional[str] = None,
    ) -> None:
        try:
            with self._db.session() as session:
                report = self._require(session, report_id)
                report.status = status.value
                if error_message is not None:
                    report.error_message = error_message

        except db_exc.DatabaseError as e:
            raise map_report_error(e, context=f"Setting status of report {report_id}") from e

    # ==========================================
    # READS
    # ==========================================

    def get_report(self, report_id: uuid.UUID) -> Optional[Report]:
        try:
            with self._db.session() as session:
                report = session.get(ResearchReportModel, report_id)
                if report is None:
                    return None
                return self._to_report(session, report)

        except db_exc.DatabaseError as e:
            raise map_report_error(e, context=f"Loading report {report_id}") from e

    def list_reports(self, session_id: Optional[uuid.UUID] = None, limit: int = 50) -> List[ReportSummary]:
        try:
            with self._db.session() as session:
                stmt = select(ResearchReportModel).order_by(ResearchReportModel.created_at.desc()).limit(limit)
                if session_id is not None:
                    stmt = stmt.where(ResearchReportModel.session_id == session_id)

                reports = session.execute(stmt).scalars().all()
                if not reports:
                    return []

                ids = [r.id for r in reports]
                section_counts = dict(session.execute(
                    select(ResearchReportSectionModel.report_id, func.count())
                    .where(ResearchReportSectionModel.report_id.in_(ids))
                    .group_by(ResearchReportSectionModel.report_id)
                ).all())
                evidence_counts = dict(session.execute(
                    select(ResearchReportEvidenceModel.report_id, func.count())
                    .where(ResearchReportEvidenceModel.report_id.in_(ids))
                    .group_by(ResearchReportEvidenceModel.report_id)
                ).all())

                return [
                    ReportSummary(
                        id=r.id,
                        topic=r.topic,
                        status=ReportStatus(r.status),
                        session_id=r.session_id,
                        section_count=section_counts.get(r.id, 0),
                        evidence_count=evidence_counts.get(r.id, 0),
                        markdown_path=r.markdown_path,
                        created_at=r.created_at,
                        completed_at=r.completed_at,
                    )
                    for r in reports
                ]

        except db_exc.DatabaseError as e:
            raise map_report_error(e, context="Listing reports") from e

    # ==========================================
    # INTERNALS
    # ==========================================

    @staticmethod
    def _require(session, report_id: uuid.UUID) -> ResearchReportModel:
        report = session.get(ResearchReportModel, report_id)
        if report is None:
            raise ReportNotFoundError(f"Report {report_id} does not exist.")
        return report

    @staticmethod
    def _write_plan_rows(session, plan: ResearchPlan) -> None:
        for section in plan.sections:
            session.add(ResearchReportSectionModel(
                report_id=plan.report_id,
                section_key=section.key,
                title=section.title,
                order_index=section.order_index,
                brief=section.brief,
                status=SectionStatus.PENDING.value,
                evidence_doc_numbers=list(section.evidence_doc_numbers),
            ))
        for question in plan.sub_questions:
            session.add(ResearchReportSubQuestionModel(
                id=question.id,
                report_id=plan.report_id,
                section_key=question.section_key,
                question_text=question.text,
                order_index=question.order_index,
            ))

    @staticmethod
    def _to_report(session, report: ResearchReportModel) -> Report:
        section_rows = session.execute(
            select(ResearchReportSectionModel)
            .where(ResearchReportSectionModel.report_id == report.id)
            .order_by(ResearchReportSectionModel.order_index)
        ).scalars().all()

        question_rows = session.execute(
            select(ResearchReportSubQuestionModel)
            .where(ResearchReportSubQuestionModel.report_id == report.id)
            .order_by(ResearchReportSubQuestionModel.order_index)
        ).scalars().all()

        evidence_rows = session.execute(
            select(ResearchReportEvidenceModel)
            .where(ResearchReportEvidenceModel.report_id == report.id)
            .order_by(ResearchReportEvidenceModel.doc_number)
        ).scalars().all()

        sub_questions = [
            SubQuestion(id=q.id, text=q.question_text, section_key=q.section_key, order_index=q.order_index)
            for q in question_rows
        ]

        outline = [
            OutlineSection(
                key=s.section_key,
                title=s.title,
                order_index=s.order_index or 0,
                brief=s.brief or "",
                sub_question_ids=[q.id for q in sub_questions if q.section_key == s.section_key],
                evidence_doc_numbers=list(s.evidence_doc_numbers or []),
            )
            for s in section_rows
        ]

        plan = ResearchPlan(
            report_id=report.id,
            topic=report.topic,
            scope_note=report.scope_note or "",
            sections=outline,
            sub_questions=sub_questions,
            revision=report.plan_revision or 1,
        )

        sections = [
            ReportSection(
                key=s.section_key,
                title=s.title,
                order_index=s.order_index or 0,
                status=SectionStatus(s.status or SectionStatus.PENDING.value),
                markdown_body=s.markdown_body or "",
                cited_doc_numbers=list(s.cited_doc_numbers or []),
                written_at=s.written_at,
            )
            for s in section_rows
        ]

        evidence = [
            EvidenceEntry(
                doc_number=e.doc_number,
                chunk_id=e.chunk_id,
                source_id=e.source_id,
                source_name=e.source_name,
                start_sec=e.start_sec,
                end_sec=e.end_sec,
                speaker_labels=list(e.speaker_labels or []),
                digest=e.digest or "",
                origin=EvidenceOrigin(e.origin or EvidenceOrigin.FAN_OUT.value),
                found_by=e.found_by,
                rerank_score=e.rerank_score,
                scrubbed=bool(e.scrubbed),
            )
            for e in evidence_rows
        ]

        corpus = None
        if report.corpus_digest:
            try:
                corpus = CorpusDigest.model_validate(report.corpus_digest)
            except Exception as e:
                # A digest written by an older build should not make the whole report unreadable.
                logger.warning(f"Could not read the stored corpus digest of report {report.id}: {e}")

        # Columns declared with server_default are None on the Python side until the database fills
        # them, which has not happened yet for a row created in this same transaction. Defaulting here
        # keeps a just-created report readable without depending on flush/RETURNING behaviour.
        return Report(
            id=report.id,
            topic=report.topic,
            status=ReportStatus(report.status),
            plan=plan,
            sections=sections,
            evidence=evidence,
            session_id=report.session_id,
            executive_summary=report.executive_summary or "",
            markdown_path=report.markdown_path,
            corpus=corpus,
            indexed_source_count=report.indexed_source_count or 0,
            swept_source_count=report.swept_source_count or 0,
            scrub_enabled=bool(report.scrub_enabled),
            mode=ReportMode(report.mode or ReportMode.DEEP.value),
            rounds_used=report.rounds_used or 1,
            error_message=report.error_message,
            created_at=report.created_at,
            updated_at=report.updated_at,
            completed_at=report.completed_at,
        )
