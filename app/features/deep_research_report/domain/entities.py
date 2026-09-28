# Path: app/features/deep_research_report/domain/entities.py

import uuid
from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.core.anchors.domain.enums import AnchorKind
from app.features.deep_research_report.domain.enums import (
    EvidenceOrigin, ReportMode, ReportStatus, SectionStatus,
)


class IndexedSource(BaseModel):
    """
    A source that actually has rows in `vector_chunks`. Distinct from storage's `Source`, which lists what
    was *ingested* — a file can be ingested and still have nothing indexed. Report mode needs the indexed
    set, both to tell the planner what material exists and to know what the sweep is allowed to miss.
    """
    source_id: uuid.UUID
    name: str
    chunk_count: int
    first_start_sec: float = 0.0
    last_end_sec: float = 0.0
    # A document is described by its page count, a recording by its length
    anchor_kind: AnchorKind = AnchorKind.TIME
    page_count: Optional[int] = None

    model_config = ConfigDict(frozen=True)


class CorpusDigest(BaseModel):
    """What the planner is shown about the corpus before it proposes anything."""
    indexed_source_count: int
    indexed_chunk_count: int
    sources: List[IndexedSource] = Field(default_factory=list)
    # Real excerpts from a grounding search on the raw topic, so the outline is proposed against the
    # corpus's actual vocabulary rather than the model's expectations of the topic.
    sample_digests: List[str] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True)


class SubQuestion(BaseModel):
    """One retrieval query the plan intends to run. `text` is used verbatim as the search prompt."""
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    text: str
    section_key: str
    order_index: int = 0

    model_config = ConfigDict(frozen=True)


class OutlineSection(BaseModel):
    """
    One section of the report.

    `key` is a stable slug, not a position: it survives the user reordering or renaming sections, which is
    what lets evidence assigned in one plan revision stay attached in the next.

    `evidence_doc_numbers` is this section's citation into the evidence bank — the link the outline holds
    and the writer reads. It is assigned by the outline-optimisation pass, and an empty list is meaningful:
    it means the corpus had nothing for this section, and the section is reported as not covered rather
    than written from the model's own knowledge.
    """
    key: str
    title: str
    order_index: int
    # 1-3 sentences saying what this section must cover. Doubles as the writer's instruction and, when
    # scrubbing is enabled, as the relevance prompt for this section's evidence group.
    brief: str = ""
    sub_question_ids: List[uuid.UUID] = Field(default_factory=list)
    evidence_doc_numbers: List[int] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True)


class FollowUpQuery(BaseModel):
    """
    A search to run in the next gathering round, for a section the last round could not support.

    Written by the model during outline optimisation, where it can already see the evidence index -- so it
    can re-word a failed sub-question using vocabulary that actually appears in the corpus, which is the
    whole reason a second round finds anything the first missed.

    Deliberately NOT a `SubQuestion`: these are transient. Appending them to the plan would trip
    REPORT_MAX_SUB_QUESTIONS on the next validation, and they are not part of what the user approved.
    """
    section_key: str
    text: str

    model_config = ConfigDict(frozen=True)


class QueryAttempt(BaseModel):
    """
    One search this run has already run, and what it was worth.

    Gathering is allowed to repeat, and without a record of what was already asked it repeats itself: the
    fallback follow-up is derived deterministically from a section's title and brief, neither of which
    changes between rounds, so a section the model writes no follow-up for is re-searched with the
    identical query every round until the budget runs out. Each of those rounds costs an embedder load, a
    reranker load and a ~5.7 GB report-model load to reach a conclusion already reached.

    `chunks_added` counts only chunks NOT already in the bank, so it answers the question the next round
    actually needs answered -- "did asking this buy us anything" -- rather than how many rows came back.
    A query that scored zero is the useful signal: its wording found nothing in this corpus, which is
    exactly what the model must be told so it re-words rather than paraphrases.
    """
    text: str
    round_index: int
    chunks_added: int = 0

    model_config = ConfigDict(frozen=True)


class OutlineAssignment(BaseModel):
    """
    What one outline-optimisation pass produced: where the evidence goes, and what to search for next.

    Both come out of a single model call because the model already has the evidence index in context --
    asking separately would cost another acquisition of a 5.7 GB model to answer a question it could
    have answered for free.
    """
    sections: List["OutlineSection"] = Field(default_factory=list)
    follow_ups: List[FollowUpQuery] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True)

    @property
    def uncovered(self) -> List["OutlineSection"]:
        """Sections the gathered evidence could not support. An empty list means the outline is covered."""
        return [s for s in self.sections if not s.evidence_doc_numbers]


class ResearchPlan(BaseModel):
    """The object the user approves, and may edit before approving."""
    report_id: uuid.UUID
    topic: str
    scope_note: str = ""
    sections: List[OutlineSection] = Field(default_factory=list)
    sub_questions: List[SubQuestion] = Field(default_factory=list)
    revision: int = 1

    model_config = ConfigDict(frozen=True)

    def sub_questions_for(self, section_key: str) -> List[SubQuestion]:
        return [q for q in self.sub_questions if q.section_key == section_key]


class ResearchPlanProposal(BaseModel):
    """
    What `propose_research_plan` hands back: the plan to approve, what the corpus holds, and enough
    information for a caller to show a determinate progress bar instead of an indeterminate spinner.
    """
    report_id: uuid.UUID
    plan: ResearchPlan
    corpus: CorpusDigest
    expected_job_count: int = 0
    estimated_minutes: int = 0

    model_config = ConfigDict(frozen=True)


class EvidenceEntry(BaseModel):
    """
    One row of the evidence memory bank, and the join between a `[Doc N]` citation and a real chunk.

    `source_name` and `digest` are denormalised on purpose. Re-ingesting a file cascades its `vector_chunks`
    away, and a signed-off report has to survive that — with these stored, a citation whose chunk is gone
    still renders with a real source, timespan and excerpt instead of becoming a dead link.

    `digest` is the head of the chunk's **verbatim transcript** (`raw_text`) only, never `text_payload`:
    the generated situating context in `text_payload` improves retrieval but is not citable source material.
    """
    doc_number: int
    chunk_id: uuid.UUID
    source_id: uuid.UUID
    source_name: str
    start_sec: float
    end_sec: float
    # Document excerpts are located by page / section; their start_sec/end_sec are 0.0
    anchor_kind: AnchorKind = AnchorKind.TIME
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section_path: Optional[str] = None
    speaker_labels: List[str] = Field(default_factory=list)
    digest: str = ""
    origin: EvidenceOrigin = EvidenceOrigin.FAN_OUT
    # The sub-question that surfaced this chunk, or None for a sweep hit.
    found_by: Optional[str] = None
    rerank_score: Optional[float] = None
    scrubbed: bool = False

    model_config = ConfigDict(frozen=True)


class ReportSection(BaseModel):
    """A written section. `cited_doc_numbers` is what the model actually cited, after validation."""
    key: str
    title: str
    order_index: int
    status: SectionStatus = SectionStatus.PENDING
    markdown_body: str = ""
    cited_doc_numbers: List[int] = Field(default_factory=list)
    written_at: Optional[datetime] = None

    model_config = ConfigDict(frozen=True)


class ReportSummary(BaseModel):
    """Listing row — enough to populate a sidebar without loading a whole report."""
    id: uuid.UUID
    topic: str
    status: ReportStatus
    session_id: Optional[uuid.UUID] = None
    section_count: int = 0
    evidence_count: int = 0
    markdown_path: Optional[str] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    model_config = ConfigDict(frozen=True)


class Report(BaseModel):
    """
    A full report: its plan, its written sections, and the evidence bank they cite.

    `indexed_source_count` vs `swept_source_count` is a coverage claim the rendered report states out loud.
    A report that swept 50 of 214 sources must not read as though it covered the corpus.
    """
    id: uuid.UUID
    topic: str
    status: ReportStatus
    plan: ResearchPlan
    sections: List[ReportSection] = Field(default_factory=list)
    evidence: List[EvidenceEntry] = Field(default_factory=list)
    session_id: Optional[uuid.UUID] = None
    executive_summary: str = ""
    markdown_path: Optional[str] = None
    # What the corpus looked like when this report was scoped. Stored rather than recomputed: it is the
    # basis of the report's coverage claim, and must stay true even after the corpus grows.
    corpus: Optional[CorpusDigest] = None
    indexed_source_count: int = 0
    swept_source_count: int = 0
    scrub_enabled: bool = False
    mode: ReportMode = ReportMode.DEEP
    # How many gathering rounds actually ran. Stated in the report, because "not covered" means something
    # different after one round than after three.
    rounds_used: int = 1
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    model_config = ConfigDict(frozen=True)

    @property
    def evidence_by_doc_number(self) -> Dict[int, EvidenceEntry]:
        return {e.doc_number: e for e in self.evidence}

    @property
    def written_sections(self) -> List[ReportSection]:
        return [s for s in self.sections if s.status == SectionStatus.WRITTEN]

    @property
    def uncovered_sections(self) -> List[ReportSection]:
        return [s for s in self.sections if s.status == SectionStatus.SKIPPED_NO_EVIDENCE]

    @property
    def cited_doc_numbers(self) -> List[int]:
        """Every doc number the body actually cites, ascending. The summary may cite only these."""
        return sorted({n for s in self.written_sections for n in s.cited_doc_numbers})
