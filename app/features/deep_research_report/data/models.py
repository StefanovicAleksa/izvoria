# Path: app/features/deep_research_report/data/models.py

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database.data.engine import Base
from app.features.deep_research_report.domain.enums import ReportStatus


class ResearchReportModel(Base):
    """
    One research report. Every stage of a run commits here, so a crash or a cancellation leaves a row that
    says exactly how far it got -- which is what makes `resume` possible without re-running retrieval.

    `indexed_source_count` and `swept_source_count` are stored rather than recomputed: they are a claim the
    rendered report makes about its own coverage, and it must stay true even after the corpus grows.
    """
    __tablename__ = "research_reports"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # Soft link: a report started from a conversation shows up in that conversation, but a report is not a
    # chat turn and does not live or die with one.
    session_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), nullable=True, index=True)

    topic: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default=ReportStatus.AWAITING_APPROVAL.value,
                                        index=True)
    scope_note: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    plan_revision: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")

    executive_summary: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    markdown_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    corpus_digest: Mapped[dict] = mapped_column(JSONB, server_default="{}")
    indexed_source_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    swept_source_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    scrub_enabled: Mapped[bool] = mapped_column(nullable=False, server_default=text("false"))
    # Which mode produced this report, and how many gathering rounds actually ran. Stored because
    # "not covered" means something different after one round than after three.
    mode: Mapped[str] = mapped_column(String, nullable=False, server_default="deep")
    rounds_used: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")
    # The language the report is written in ("sr"). A report reopened or resumed after the active language
    # changed is still rendered and finished in this one.
    language: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                 default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                 default=lambda: datetime.now(timezone.utc),
                                                 onupdate=lambda: datetime.now(timezone.utc))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("idx_research_reports_session", "session_id", "created_at"),
    )


class ResearchReportSectionModel(Base):
    """
    One outline section, written or not.

    `section_key` rather than position is the identity, so a user reordering the outline between revisions
    does not detach a section from the evidence assigned to it.
    """
    __tablename__ = "research_report_sections"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    report_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("research_reports.id", ondelete="CASCADE"), nullable=False, index=True
    )

    section_key: Mapped[str] = mapped_column(String, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    order_index: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    brief: Mapped[str] = mapped_column(Text, nullable=False, server_default="")

    status: Mapped[str] = mapped_column(String, nullable=False, server_default="pending")
    markdown_body: Mapped[str] = mapped_column(Text, nullable=False, server_default="")

    # What the outline assigned, and what the writer actually cited. Both are kept: the gap between them
    # is the honest measure of how much of a section's evidence the model actually used.
    evidence_doc_numbers: Mapped[List[int]] = mapped_column(
        ARRAY(Integer), nullable=False, server_default=text("'{}'::integer[]")
    )
    cited_doc_numbers: Mapped[List[int]] = mapped_column(
        ARRAY(Integer), nullable=False, server_default=text("'{}'::integer[]")
    )

    written_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        UniqueConstraint("report_id", "section_key", name="uq_research_report_section_key"),
        Index("idx_research_report_sections_order", "report_id", "order_index"),
    )


class ResearchReportEvidenceModel(Base):
    """
    One row of the evidence memory bank -- the join between a `[Doc N]` citation and a real chunk.

    `chunk_id` is deliberately NOT a foreign key to `vector_chunks`. Chunks cascade from `sources`, so
    re-ingesting a file deletes them, and a signed-off report must survive that: the denormalised
    `source_name`, timespan, speakers and `digest` let a citation still render with real provenance after
    its chunk is gone, instead of the report quietly losing a reference someone relied on.
    """
    __tablename__ = "research_report_evidence"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    report_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("research_reports.id", ondelete="CASCADE"), nullable=False, index=True
    )

    doc_number: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    source_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    source_name: Mapped[str] = mapped_column(Text, nullable=False, server_default="")

    start_sec: Mapped[float] = mapped_column(Float, nullable=False, server_default="0")
    end_sec: Mapped[float] = mapped_column(Float, nullable=False, server_default="0")
    anchor_kind: Mapped[str] = mapped_column(String, nullable=False, server_default="time")
    page_start: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    page_end: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    section_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    speaker_labels: Mapped[List[str]] = mapped_column(JSONB, nullable=False, default=list)
    digest: Mapped[str] = mapped_column(Text, nullable=False, server_default="")

    origin: Mapped[str] = mapped_column(String, nullable=False, server_default="fan_out")
    found_by: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    rerank_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    scrubbed: Mapped[bool] = mapped_column(nullable=False, server_default=text("false"))

    __table_args__ = (
        UniqueConstraint("report_id", "doc_number", name="uq_research_report_evidence_doc_number"),
        UniqueConstraint("report_id", "chunk_id", name="uq_research_report_evidence_chunk"),
    )


class ResearchReportSubQuestionModel(Base):
    """The queries the plan intends to run, kept so a resumed or reopened report explains its own coverage."""
    __tablename__ = "research_report_sub_questions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    report_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("research_reports.id", ondelete="CASCADE"), nullable=False, index=True
    )

    section_key: Mapped[str] = mapped_column(String, nullable=False)
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    order_index: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    chunks_found: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
