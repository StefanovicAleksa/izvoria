# Canvas: engine_interfaces_v3
# Path: app/engine/domain/interfaces.py

import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from app.engine.domain.enums import EngineState
from app.engine.domain.entities import IngestionReport
from app.features.rag_retrieval.domain.entities import RetrievedChunk, SearchQuery, SearchResult
from app.features.storage.domain.entities import Source
from app.features.qa_generation.domain.entities import QAResponse
from app.orchestration.domain.entities import ReasonedChatResponse
from app.features.deep_research_report.domain.entities import (
    Report, ReportSummary, ResearchPlan, ResearchPlanProposal,
)
from app.features.search_history.domain.entities import SearchRecord, SearchRecordSummary
from app.features.session_memory.domain.entities import SessionSummary, SessionTurn

class IOnyxEngine(ABC):
    """
    The unified public Facade for the Onyx AI Appliance.

    `owner_id` is the signed-in user on the hosted multi-user API: their turns, searches and reports are recorded
    under it and every history read is limited to it. The appliance never passes one, and history stays shared.
    """
    @abstractmethod
    def start(self) -> None:
        """Boots the engine, initializes DB connection pools, and prepares VRAM."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Gracefully shuts down background workers, clears VRAM, and closes DB connections."""
        pass

    @abstractmethod
    def get_state(self) -> EngineState:
        """Returns the current operational state of the appliance."""
        pass

    @property
    @abstractmethod
    def active_language(self) -> str:
        """The language the next ingestion batch is indexed in and answers are written in ("sr")."""
        pass

    @abstractmethod
    def require_idle(self) -> None:
        """Raises EngineBusyError while ingestion, a report, a live call or booting holds the engine."""
        pass

    @abstractmethod
    def busy_snapshot(self) -> Dict[str, int]:
        """What the engine is doing right now: {"inflight": live calls, "reports_running": reports}."""
        pass

    @abstractmethod
    def ingest_videos(self, target_path: str) -> IngestionReport:
        """
        Runs the full Batch Media Pipeline (Audio + Vision) on a directory.
        Clears VRAM before starting, and preloads search models when finished.
        """
        pass

    @abstractmethod
    def ingest_audios(self, target_path: str) -> IngestionReport:
        """
        Runs the Batch Audio Pipeline on a directory, bypassing visual processing.
        Clears VRAM before starting, and preloads search models when finished.
        """
        pass

    @abstractmethod
    def ingest_documents(self, target_path: str) -> IngestionReport:
        """
        Runs the Batch Document Pipeline (PDF / DOCX, OCR for scanned pages) on a directory.
        Clears VRAM before starting, and preloads search models when finished.
        """
        pass

    @abstractmethod
    def ingest_images(self, target_path: str) -> IngestionReport:
        """
        Runs the Batch Image Pipeline (vision model description, checked by OCR) on a directory.
        Clears VRAM before starting, and preloads search models when finished.
        """
        pass

    @abstractmethod
    def search_standard(self, query: SearchQuery, owner_id: Optional[str] = None) -> SearchResult:
        """
        Executes a fast, standard RAG hybrid search against the Onyx Vault.
        """
        pass

    @abstractmethod
    def search_verified(self, query: SearchQuery, owner_id: Optional[str] = None) -> SearchResult:
        """
        Executes a high-precision RAG search followed immediately by an SLM
        hallucination scrubber to drop irrelevant context in VRAM.
        """
        pass

    @abstractmethod
    def chat(self, query: SearchQuery, session_id: Optional[uuid.UUID] = None,
             owner_id: Optional[str] = None) -> QAResponse:
        """
        Executes an end-to-end Chat pipeline:
        Hybrid Search -> SLM Semantic Scrubbing -> LLM Final Generation.
        With a session_id the question is a turn of that conversation: a follow-up is searched in standalone form,
        the answer model sees the earlier turns and the evidence they showed, and the turn is recorded.
        """
        pass

    @abstractmethod
    def chat_reasoned(self, session_id: uuid.UUID, query: SearchQuery,
                      owner_id: Optional[str] = None) -> ReasonedChatResponse:
        """
        Multi-pass reasoning chat: a Qwen3.5-9B Reasoner loops against retrieval/rerank/scrub
        until ready to answer or in need of clarification, then the answer model drafts an answer and
        immediately self-critiques it (same model acquisition), looping back for more evidence
        if under-supported. Bounded by REASONING_MAX_ITERATIONS / REASONING_MAX_CRITIQUE_RETRIES.
        Shares the session's conversation with chat(): it reads the earlier turns and their evidence, and
        records its own turn.
        """
        pass

    @abstractmethod
    def propose_research_plan(
        self,
        topic: str,
        session_id: Optional[uuid.UUID] = None,
        source_id_filter: Optional[uuid.UUID] = None,
        owner_id: Optional[str] = None
    ) -> ResearchPlanProposal:
        """
        Deep Research Report mode, stage one: search the corpus once, propose an outline and the
        sub-questions to research, and stop for approval.

        Two calls rather than one, because the run that follows takes tens of minutes and a misread topic
        is cheap to correct here. The returned plan is editable -- reorder, rename, delete or add sections
        and sub-questions -- and comes back to execute_research_report as `approved_plan`.
        """
        pass

    @abstractmethod
    def execute_research_report(self, report_id: uuid.UUID, approved_plan: ResearchPlan) -> Report:
        """
        Deep Research Report mode, stage two: sweep the corpus, assign the evidence, write the report
        section by section, and render it to Markdown.

        Holds the GPU exclusively (EngineState.RESEARCHING), so chat and search are rejected with
        EngineBusyError until it finishes or is cancelled. Every claim carries a [Doc N] citation
        validated against the evidence bank; a section the corpus cannot support is reported as not
        covered rather than written.
        """
        pass

    @abstractmethod
    def execute_iterative_research_report(self, report_id: uuid.UUID, approved_plan: ResearchPlan) -> Report:
        """
        Exhaustive mode: the same run as execute_research_report, but gathering repeats while sections
        come back unsupported.

        A section with no evidence does not distinguish "the corpus does not cover this" from "the
        sub-question was worded badly". This mode re-asks, using wording the model can now see in the
        gathered evidence, up to REPORT_MAX_GATHER_ROUNDS times, and stops as soon as a round turns up
        nothing new.
        """
        pass

    @abstractmethod
    def resume_research_report(self, report_id: uuid.UUID) -> Report:
        """
        Finishes a cancelled or failed report from its persisted evidence bank, writing only the sections
        still pending. Loads no retrieval model: the bank already holds everything that was gathered.
        """
        pass

    @abstractmethod
    def cancel_research_report(self, report_id: uuid.UUID) -> bool:
        """
        Asks a running report to stop, returning whether one was actually running.

        Safe from any thread. The run stops at the next section boundary -- up to about ninety seconds --
        because the generation call it is inside cannot be interrupted. Everything written by then is kept
        and the report stays resumable.
        """
        pass

    @abstractmethod
    def get_report(self, report_id: uuid.UUID, owner_id: Optional[str] = None) -> Optional[Report]:
        """A stored report with its plan, sections and evidence bank (None if not `owner_id`'s). No model, no lock."""
        pass

    @abstractmethod
    def rename_research_report(self, report_id: uuid.UUID, title: str,
                               owner_id: Optional[str] = None) -> Optional[Report]:
        """Sets a report's title ("" clears it). None if there is no such report, or it is not `owner_id`'s."""
        pass

    @abstractmethod
    def list_reports(self, session_id: Optional[uuid.UUID] = None, limit: int = 50,
                     owner_id: Optional[str] = None) -> List[ReportSummary]:
        """Stored reports, newest first, optionally scoped to one conversation."""
        pass

    @abstractmethod
    def get_chunks(self, chunk_ids: List[uuid.UUID]) -> List[RetrievedChunk]:
        """
        The verbatim text and resolved clip paths behind a set of chunk ids.
        A citation carries ids and timestamps but no text, so a UI showing a cited excerpt beside
        its clip reads the chunks back through here. A plain indexed read: no model, no VRAM.
        """
        pass

    @abstractmethod
    def list_sources(self) -> List[Source]:
        """Everything ingested into the vault, newest first, each with its file row attached."""
        pass

    # History: database reads, no model and no engine lock, so they work while ingestion or a report runs.

    @abstractmethod
    def list_conversations(self, limit: int = 200, owner_id: Optional[str] = None) -> List[SessionSummary]:
        """Chat conversations, most recently active first."""
        pass

    @abstractmethod
    def get_conversation(self, session_id: uuid.UUID, owner_id: Optional[str] = None) -> List[SessionTurn]:
        """Every turn of a conversation, oldest first."""
        pass

    @abstractmethod
    def delete_conversation(self, session_id: uuid.UUID, owner_id: Optional[str] = None) -> int:
        """Forgets a conversation's turns; returns how many there were."""
        pass

    @abstractmethod
    def list_searches(self, limit: int = 200, owner_id: Optional[str] = None) -> List[SearchRecordSummary]:
        """Past standard and verified searches, newest first."""
        pass

    @abstractmethod
    def get_search(self, record_id: uuid.UUID, owner_id: Optional[str] = None) -> Optional[SearchRecord]:
        """A past search with its ranked chunk ids and scores."""
        pass

    @abstractmethod
    def delete_search(self, record_id: uuid.UUID, owner_id: Optional[str] = None) -> bool:
        """Forgets a past search; says whether it existed."""
        pass
