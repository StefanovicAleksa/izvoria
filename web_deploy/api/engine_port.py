"""
What the HTTP layer needs from the appliance, as a port.

Routes, mapping and the runner talk to an IEngineHost and never import `app.*` themselves; the one implementation
that does (engine_host.EngineHost) is wired in at startup. That keeps the whole API testable against a fake host in
a light venv, with no torch, GPU, database or weights, exactly as the desktop app's StubEngineGateway does.
"""

import threading
import time
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


class ApplianceIngesting(RuntimeError):
    """Live work refused while ingestion holds the whole GPU (dev hardware, GPU_SHARING=exclusive)."""

    def __init__(self) -> None:
        super().__init__("The appliance is ingesting files and has handed the GPU to it. Chat, search and reports "
                         "resume as soon as the ingestion queue is empty; history stays readable meanwhile.")


class IEngineHost(ABC):

    # ---------- lifecycle ----------

    @abstractmethod
    def start(self) -> None:
        """Loads the shared models (pinned and external). Blocks until they are resident."""

    @abstractmethod
    def stop(self) -> None:
        pass

    @property
    @abstractmethod
    def ready(self) -> bool:
        pass

    @abstractmethod
    def busy(self) -> Dict[str, int]:
        """Live calls and reports in flight across every firm: {"inflight": n, "reports_running": m}."""

    # ---------- handing the GPU to ingestion (dev hardware) ----------
    # On a card too small for the live models and an ingestion batch at once, the ingestion worker asks the API
    # to give up the GPU (pause), runs its jobs, and gives it back (resume). Every piece of model work is done
    # inside `live()`, which refuses while paused and lets pause() wait for whatever was already admitted.

    @property
    @abstractmethod
    def paused(self) -> bool:
        pass

    @abstractmethod
    def live(self) -> AbstractContextManager:
        """Admits one piece of model work; raises ApplianceIngesting while the GPU is handed to ingestion."""

    @abstractmethod
    def pause(self, poll_seconds: float = 1.0) -> None:
        """Refuses new model work, waits for admitted work to finish, then frees every live model."""

    @abstractmethod
    def resume(self) -> None:
        """Loads the live models again and admits model work."""

    # ---------- firms ----------

    @abstractmethod
    def engine(self, firm: str) -> Any:
        """The firm's started OnyxEngine, created (database, migrations, vault) on first use."""

    @abstractmethod
    def firm_dir(self, firm: str) -> Path:
        """The firm's directory on the box: vault/, reports/, uploads/ below it."""

    # ---------- language ----------

    @abstractmethod
    def languages(self) -> List[Dict[str, str]]:
        """[{"code": "en", "name": "English"}, ...]: the languages answers can be written in."""

    @abstractmethod
    def language_scope(self, code: str) -> AbstractContextManager:
        """Answers in `code` for the current thread's request (app.core.language.language_scope)."""

    # ---------- engine value objects and helpers ----------

    @abstractmethod
    def search_query(self, prompt: str, limit: int) -> Any:
        pass

    @abstractmethod
    def normalize_citation_tags(self, text: str, valid_numbers: Iterable[int]) -> str:
        pass

    @abstractmethod
    def format_location(self, kind: str, page_start: Optional[int], page_end: Optional[int],
                        section_path: Optional[str]) -> str:
        pass

    @abstractmethod
    def render_report(self, report: Any) -> str:
        """The report as Markdown with clickable `[Doc N](onyx://doc/N)` citations."""

    @abstractmethod
    def display_math(self, markdown: str) -> str:
        """
        Report Markdown with its formulas in the forms the web page's math plugin reads (`$$…$$`), money left as
        text. For showing only: the Markdown download keeps the writer's LaTeX as written.
        """

    @abstractmethod
    def report_pdf(self, report: Any) -> bytes:
        """The report as a typeset PDF, laid out from the stored report (never from the model)."""

    @abstractmethod
    def outline_section(self, **fields: Any) -> Any:
        pass

    @abstractmethod
    def sub_question(self, **fields: Any) -> Any:
        pass


class LiveGate:
    """
    The admission half of pause/resume, shared by the real host and test fakes. `live()` counts admitted work under
    the same lock as the paused flag, so pause() can never miss a request admitted a moment before it.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = 0
        self._paused = False

    @property
    def paused(self) -> bool:
        return self._paused

    @contextmanager
    def live(self):
        with self._lock:
            if self._paused:
                raise ApplianceIngesting()
            self._active += 1
        try:
            yield
        finally:
            with self._lock:
                self._active -= 1

    def close_and_drain(self, poll_seconds: float = 1.0) -> None:
        with self._lock:
            self._paused = True
        while True:
            with self._lock:
                if self._active == 0:
                    return
            time.sleep(poll_seconds)

    def reopen(self) -> None:
        with self._lock:
            self._paused = False
