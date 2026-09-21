# Canvas: engine_interfaces_v3
# Path: app/engine/domain/interfaces.py

import uuid
from abc import ABC, abstractmethod
from typing import Optional

from app.engine.domain.enums import EngineState
from app.engine.domain.entities import IngestionReport
from app.features.rag_retrieval.domain.entities import SearchQuery, SearchResult
from app.features.qa_generation.domain.entities import QAResponse
from app.orchestration.domain.entities import DeepResearchResponse

class IOnyxEngine(ABC):
    """
    The unified public Facade for the Onyx AI Appliance.
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
    def search_standard(self, query: SearchQuery) -> SearchResult:
        """
        Executes a fast, standard RAG hybrid search against the Onyx Vault.
        """
        pass

    @abstractmethod
    def search_verified(self, query: SearchQuery) -> SearchResult:
        """
        Executes a high-precision RAG search followed immediately by an SLM
        hallucination scrubber to drop irrelevant context in VRAM.
        """
        pass

    @abstractmethod
    def chat(self, query: SearchQuery, session_id: Optional[uuid.UUID] = None) -> QAResponse:
        """
        Executes an end-to-end Chat pipeline:
        Hybrid Search -> SLM Semantic Scrubbing -> LLM Final Generation.
        With a session_id the question is a turn of that conversation: a follow-up is searched in standalone form,
        the answer model sees the earlier turns and the evidence they showed, and the turn is recorded.
        """
        pass

    @abstractmethod
    def chat_deep_research(self, session_id: uuid.UUID, query: SearchQuery) -> DeepResearchResponse:
        """
        Multi-pass reasoning chat: a Qwen3.5-9B Reasoner loops against retrieval/rerank/scrub
        until ready to answer or in need of clarification, then the answer model drafts an answer and
        immediately self-critiques it (same model acquisition), looping back for more evidence
        if under-supported. Bounded by DEEP_RESEARCH_MAX_ITERATIONS / DEEP_RESEARCH_MAX_CRITIQUE_RETRIES.
        Shares the session's conversation with chat(): it reads the earlier turns and their evidence, and
        records its own turn.
        """
        pass

    @abstractmethod
    def interactive_cli(self) -> None:
        """
        Launches an interactive command-line interface for operating the appliance.
        """
        pass