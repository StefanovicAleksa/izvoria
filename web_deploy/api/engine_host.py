"""
The only module of the hosted API that imports `app.*`.

One process serves every firm. It holds ONE model kit (the orchestrator plus the live models: BGE-M3 and the reranker
pinned in this process, the Qwen roles served by llama-server) and builds one IzvoriaEngine per firm on first use, each
with the firm's own database, vault and reports directory. Firm engines share the kit, so a new firm costs a database
connection pool, not a model load.

Imports of `app.*` happen in `start()`, not at module import, for the same reason as the desktop gateway's: they pull
in torch and read ENV at import time.
"""

import logging
import threading
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from web_deploy.api.config import ApiSettings
from web_deploy.api.engine_port import IEngineHost, LiveGate
from web_deploy.common.tenancy import database_name, firm_root, validate_slug

logger = logging.getLogger(__name__)


class EngineHost(IEngineHost):
    def __init__(self, settings: ApiSettings, progress_listener: Callable[..., None]) -> None:
        self._settings = settings
        self._progress_listener = progress_listener
        self._kit = None
        self._engines: Dict[str, Any] = {}
        self._engines_lock = threading.Lock()
        # One lock per firm: creating a firm's database must not block requests for every other firm
        self._firm_locks: Dict[str, threading.Lock] = {}
        self._app: Dict[str, Any] = {}
        self._ready = False
        self._gate = LiveGate()
        self._handoff_lock = threading.Lock()

    # ---------- lifecycle ----------

    def start(self) -> None:
        from app.core.anchors.utils.formatting import format_location
        from app.core.config import settings
        from app.core.database.service.manager import DatabaseManager
        from app.core.language.service.language_api import get_supported_language_packs, language_scope
        from app.engine.service.bootstrap import IzvoriaBootstrapper
        from app.features.deep_research_report.domain.entities import OutlineSection, SubQuestion
        from app.features.deep_research_report.utils import markdown_renderer
        from app.features.report_export.service.report_export_api import ReportExportAPI
        from app.features.report_export.utils.math_text import normalize_for_display
        from app.features.qa_generation.utils.citation_parser import normalize_citation_tags
        from app.features.rag_retrieval.domain.entities import SearchQuery

        if not settings.ENGINE_CONCURRENT_LIVE:
            raise RuntimeError(
                "The hosted API serves many users at once and needs ENGINE_CONCURRENT_LIVE=true, with every live "
                "model pinned or external (see web_deploy/provision/api.env.example)."
            )

        self._app = dict(
            settings=settings, DatabaseManager=DatabaseManager, IzvoriaBootstrapper=IzvoriaBootstrapper,
            SearchQuery=SearchQuery, normalize_citation_tags=normalize_citation_tags,
            format_location=format_location, markdown_renderer=markdown_renderer,
            OutlineSection=OutlineSection, SubQuestion=SubQuestion, language_scope=language_scope,
            ReportExportAPI=ReportExportAPI, normalize_math=normalize_for_display,
            language_packs=get_supported_language_packs,
        )
        if settings.LLM_BACKEND == "llama_server":
            self._wait_for_llama_server(settings.LLM_SERVER_URL)
        self._kit = IzvoriaBootstrapper.build_model_kit()
        warmed = self._kit.warm_up()
        logger.info(f"Shared models resident: {', '.join(t.value for t in warmed)}")
        self._ready = True

    @staticmethod
    def _wait_for_llama_server(url: str, timeout_s: float = 600.0) -> None:
        """llama-server starts beside the API (supervisord, or the dev launcher) and takes a minute to load the 9B."""
        import time
        from app.core.model_manager.data.llama_server_chat_model import LlamaServerChatModel
        probe = LlamaServerChatModel(url, 0, 10.0)
        deadline = time.monotonic() + timeout_s
        while not probe.health():
            if time.monotonic() > deadline:
                raise RuntimeError(f"llama-server at {url} did not come up within {timeout_s / 60:.0f} minutes.")
            time.sleep(2.0)
        logger.info(f"llama-server at {url} is up.")

    def stop(self) -> None:
        self._ready = False
        with self._engines_lock:
            engines, self._engines = list(self._engines.values()), {}
        for engine in engines:
            try:
                engine.stop()
            except Exception as exc:
                logger.error(f"Stopping a firm's engine failed: {exc}")
        if self._kit is not None:
            self._kit.shutdown()
            self._kit = None

    @property
    def ready(self) -> bool:
        return self._ready

    # ---------- handing the GPU to ingestion ----------

    @property
    def paused(self) -> bool:
        return self._gate.paused

    def live(self):
        return self._gate.live()

    def pause(self, poll_seconds: float = 1.0) -> None:
        with self._handoff_lock:
            if self._gate.paused:
                return
            logger.info("Handing the GPU to ingestion: refusing new questions, waiting for those in flight...")
            self._gate.close_and_drain(poll_seconds)
            # Pinned models are freed and the external clients dropped; llama-server itself is stopped by the
            # ingestion worker, which owns it on dev hardware
            self._kit.orchestrator.shutdown_all({})
            logger.info("Live models released; the GPU is the ingestion worker's.")

    def resume(self) -> None:
        with self._handoff_lock:
            if not self._gate.paused:
                return
            warmed = self._kit.warm_up()
            self._gate.reopen()
            logger.info(f"GPU back from ingestion; live models resident again: {', '.join(t.value for t in warmed)}")

    def busy(self) -> Dict[str, int]:
        with self._engines_lock:
            engines = list(self._engines.values())
        totals = {"inflight": 0, "reports_running": 0}
        for engine in engines:
            for key, value in engine.busy_snapshot().items():
                totals[key] = totals.get(key, 0) + value
        return totals

    # ---------- firms ----------

    def firm_dir(self, firm: str) -> Path:
        return firm_root(self._settings.tenants_root, firm)

    def engine(self, firm: str) -> Any:
        firm = validate_slug(firm)
        if not self._ready:
            raise RuntimeError("The appliance is still starting.")
        with self._engines_lock:
            engine = self._engines.get(firm)
            if engine is not None:
                return engine
            lock = self._firm_locks.setdefault(firm, threading.Lock())
        with lock:
            with self._engines_lock:
                engine = self._engines.get(firm)
            if engine is None:
                engine = self._create_engine(firm)
                with self._engines_lock:
                    self._engines[firm] = engine
            return engine

    def _create_engine(self, firm: str) -> Any:
        settings = self._app["settings"]
        name = database_name(firm)
        self._ensure_database(name)
        root = self.firm_dir(firm)
        for sub in ("vault", "reports", "uploads"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        url = (f"postgresql+psycopg2://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
               f"@{settings.POSTGRES_SERVER}:{settings.POSTGRES_PORT}/{name}")
        database = self._app["DatabaseManager"](url=url, pool_size=self._settings.firm_pool_size,
                                                max_overflow=self._settings.firm_pool_overflow)
        try:
            engine = self._app["IzvoriaBootstrapper"].build(
                progress_listener=self._progress_listener, database=database,
                vault_dir=root / "vault", reports_dir=root / "reports", kit=self._kit,
            )
            engine.start()  # migrates the firm's database to head
        except Exception:
            # Not cached, so the next request builds a fresh engine and pool; this one's connections must go now
            database.dispose()
            raise
        logger.info(f"Firm '{firm}' is online ({name}).")
        return engine

    def _ensure_database(self, name: str) -> None:
        from web_deploy.api.firm_db import ensure_firm_database
        settings = self._app["settings"]
        ensure_firm_database(name, settings.POSTGRES_SERVER, settings.POSTGRES_PORT, settings.POSTGRES_USER,
                             settings.POSTGRES_PASSWORD, self._settings.admin_database)

    # ---------- language ----------

    def languages(self) -> List[Dict[str, str]]:
        return [{"code": p.code, "name": p.native_name} for p in self._app["language_packs"]()]

    def language_scope(self, code: str) -> AbstractContextManager:
        return self._app["language_scope"](code)

    # ---------- value objects and helpers ----------

    def search_query(self, prompt: str, limit: int) -> Any:
        return self._app["SearchQuery"](prompt=prompt, limit=limit)

    def normalize_citation_tags(self, text: str, valid_numbers: Iterable[int]) -> str:
        return self._app["normalize_citation_tags"](text, 0, set(valid_numbers))

    def format_location(self, kind: str, page_start: Optional[int], page_end: Optional[int],
                        section_path: Optional[str]) -> str:
        return self._app["format_location"](kind, page_start=page_start, page_end=page_end,
                                            section_path=section_path)

    def render_report(self, report: Any) -> str:
        return self._app["markdown_renderer"].render(report, link_citations=True)

    def display_math(self, markdown: str) -> str:
        return self._app["normalize_math"](markdown)

    def report_pdf(self, report: Any) -> bytes:
        return self._app["ReportExportAPI"]().pdf(report)

    def outline_section(self, **fields: Any) -> Any:
        return self._app["OutlineSection"](**fields)

    def sub_question(self, **fields: Any) -> Any:
        return self._app["SubQuestion"](**fields)
