# Canvas: engine_interfaces_v1
# Path: app/engine/domain/interfaces.py

from abc import ABC, abstractmethod

from app.engine.domain.enums import EngineState
from app.engine.domain.entities import IngestionReport
from app.features.rag_retrieval.domain.entities import SearchQuery, SearchResult

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
    def ingest(self, target_path: str) -> IngestionReport:
        """
        Scans a path and automatically routes files (audio/video) to the correct pipelines.
        BLOCKING CALL: Returns when the entire directory is processed.
        """
        pass

    @abstractmethod
    def search(self, query: SearchQuery) -> SearchResult:
        """
        Executes a RAG hybrid search against the Onyx Vault, returning contextual chunks
        and pointers to the physical media clips.
        """
        pass