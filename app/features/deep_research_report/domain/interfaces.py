# Path: app/features/deep_research_report/domain/interfaces.py

import uuid
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from app.core.model_manager.domain.entities import LoadedModel, ModelManifest
from app.core.model_manager.domain.interfaces import IModelLoader
from app.features.deep_research_report.domain.entities import (
    CorpusDigest, EvidenceEntry, OutlineAssignment, OutlineSection, Report, ReportSection,
    ReportSummary, ResearchPlan,
)
from app.features.deep_research_report.domain.enums import ReportMode, ReportStatus


class IReportLlmAdapter(IModelLoader, ABC):
    """
    Contract for the report model (Qwen3.5-9B), covering BOTH the planner and the writer roles.

    They are one interface, one adapter and one ModelType deliberately. Residency is keyed on
    (model_type, model_path), so splitting planning and writing across two ModelTypes would reload the same
    ~5.7 GB GGUF on every alternation between them -- the exact cost that already makes switching between
    QA_GENERATION and QA_REASONER expensive. Two adapters sharing one ModelType is worse: the orchestrator
    tracks the loader that performed the load, and releasing with a different one is undefined.

    Keeping them together is what lets a whole report run hold a single acquisition across outline
    optimisation and every section write. Do not split this in two.
    """

    @abstractmethod
    def load(self, manifest: ModelManifest) -> LoadedModel:
        """Loads the report LLM into VRAM."""
        pass

    @abstractmethod
    def unload(self, model: LoadedModel) -> bool:
        """Safely removes the LLM from VRAM and triggers garbage collection."""
        pass

    @abstractmethod
    def count_tokens(self, model: LoadedModel, text: str) -> int:
        """Number of tokens the loaded model's tokenizer produces for text."""
        pass

    @abstractmethod
    def get_section_token_budget(self, model: LoadedModel, section: OutlineSection, topic: str,
                                 has_documents: bool = False) -> int:
        """
        Tokens left for a section's evidence block once its prompt and the generated section are counted.
        Capped by REPORT_SECTION_EVIDENCE_TOKENS: the window is large, but prefill is the cost driver.
        """
        pass

    @abstractmethod
    def scope(self, model: LoadedModel, topic: str, corpus: CorpusDigest) -> ResearchPlan:
        """
        Proposes the outline and the sub-questions to research, given what the corpus actually holds.

        The corpus digest is the first grounding checkpoint of the whole run: the planner sees real source
        names and real excerpts, so it cannot propose a section the corpus has nothing on before any GPU
        time goes into writing. Returns a plan with `revision=1` and no evidence assigned yet.
        """
        pass

    @abstractmethod
    def optimize_outline(
        self,
        model: LoadedModel,
        plan: ResearchPlan,
        evidence_digest: str,
        asked_block: str = "",
        has_documents: bool = False,
    ) -> OutlineAssignment:
        """
        Re-assigns the outline against the evidence actually gathered, returning sections carrying their
        `evidence_doc_numbers`. Sections it can support nothing with come back with an empty list, which the
        caller reports as uncovered rather than writing.

        The same call also returns a follow-up query per uncovered section, because the model already has
        the evidence index in context: it can re-word a sub-question that failed using vocabulary that
        actually appears in the corpus. Asking separately would cost another 5.7 GB acquisition to answer
        a question this call could answer for free.

        `evidence_digest` is the whole bank rendered one line per entry (doc number, source, timespan,
        excerpt head) -- compact enough that the entire bank fits one call, which is what makes a single
        model acquisition cover planning and writing both.

        `asked_block` is the searches earlier rounds already ran and what each one returned, empty on the
        first round and in deep mode. It exists so the follow-up is a genuinely new question: the model
        cannot re-word a query it cannot see, and a query marked as having added nothing tells it which
        vocabulary the archive does not contain.
        """
        pass

    @abstractmethod
    def write_section(
        self,
        model: LoadedModel,
        topic: str,
        section: OutlineSection,
        evidence_block: str,
        allowed_doc_numbers: List[int],
        outline_context: str,
        has_documents: bool = False,
    ) -> str:
        """
        Writes one section as Markdown, citing `[Doc N]` from `allowed_doc_numbers` only.

        `evidence_block` holds verbatim transcript for this section's chunks and nothing else; the caller
        discards it before the next section, so no section ever sees another's evidence. `outline_context`
        is the surrounding headings, so the section knows what it should not duplicate.
        """
        pass

    @abstractmethod
    def write_summary(
        self,
        model: LoadedModel,
        topic: str,
        section_digest: str,
        allowed_doc_numbers: List[int],
    ) -> str:
        """
        Writes the executive summary from the written sections, not from the evidence.

        The summary is the highest hallucination surface in a report because it is the one place that
        synthesises across sections, so it is restricted twice: it reads only section prose, and it may
        cite only numbers the body already cited.
        """
        pass


class IReportRepository(ABC):
    """Persistence for reports, their plans, their sections and their evidence bank."""

    @abstractmethod
    def create_report(
        self,
        topic: str,
        plan: ResearchPlan,
        corpus: CorpusDigest,
        session_id: Optional[uuid.UUID] = None,
        language: Optional[str] = None,
    ) -> Report:
        """Persists a freshly scoped report as AWAITING_APPROVAL, written in `language`."""
        pass

    @abstractmethod
    def replace_plan(self, report_id: uuid.UUID, plan: ResearchPlan) -> Report:
        """
        Stores the user's approved (possibly edited) outline, bumps the revision and moves the report to
        GATHERING. Replaces sections and sub-questions wholesale in one transaction.
        """
        pass

    @abstractmethod
    def save_evidence_bank(self, report_id: uuid.UUID, evidence: List[EvidenceEntry],
                           swept_source_count: int, scrub_enabled: bool,
                           mode: ReportMode = ReportMode.DEEP, rounds_used: int = 1) -> None:
        """
        Persists the numbered bank in one transaction and moves the report to WRITING. This is the commit
        that makes a run resumable without touching a retrieval model again.
        """
        pass

    @abstractmethod
    def save_section_assignments(self, report_id: uuid.UUID, sections: List[OutlineSection]) -> None:
        """Stores the outline-optimisation pass's per-section evidence assignments."""
        pass

    @abstractmethod
    def save_section(self, report_id: uuid.UUID, section: ReportSection) -> None:
        """Commits one finished section immediately, so a crash costs at most the section in flight."""
        pass

    @abstractmethod
    def finalize(self, report_id: uuid.UUID, executive_summary: str, markdown_path: str) -> Report:
        """Stores the summary and the rendered file's path, and marks the report COMPLETED."""
        pass

    @abstractmethod
    def set_status(self, report_id: uuid.UUID, status: ReportStatus, error_message: Optional[str] = None) -> None:
        """Records a terminal or intermediate state (CANCELLED, FAILED, ...)."""
        pass

    @abstractmethod
    def rename_report(self, report_id: uuid.UUID, title: str) -> None:
        """Sets the report's title; "" clears it, so it is shown by its request again."""
        pass

    @abstractmethod
    def set_markdown_path(self, report_id: uuid.UUID, markdown_path: str) -> None:
        """Points the report at a re-rendered file (after a rename); its status is left as it is."""
        pass

    @abstractmethod
    def set_owner(self, report_id: uuid.UUID, owner_id: str) -> None:
        """Records which user the report belongs to (the hosted API)."""
        pass

    @abstractmethod
    def get_report(self, report_id: uuid.UUID, owner_id: Optional[str] = None) -> Optional[Report]:
        """The full report: plan, sections and evidence bank."""
        pass

    @abstractmethod
    def list_reports(self, session_id: Optional[uuid.UUID] = None, limit: int = 50,
                     owner_id: Optional[str] = None) -> List[ReportSummary]:
        """Newest first. Scoped to one conversation when session_id is given."""
        pass


class IReportFileStore(ABC):
    """Writes the rendered report where a person can pick it up."""

    @abstractmethod
    def write_markdown(self, report_id: uuid.UUID, markdown: str) -> str:
        """
        Writes the report atomically and returns its absolute path. Atomically because a resume rewrites
        the same file, and a half-written report on disk is worse than none.
        """
        pass


# (report_id, markdown_path) for callers that want both without re-reading the row.
RenderedReport = Tuple[uuid.UUID, str]
