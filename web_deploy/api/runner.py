"""
Where engine calls actually run: bounded thread pools, off the event loop.

The engine is synchronous and a chat turn takes seconds to minutes, so every call runs on a worker thread, in its
own run. Live calls (chat, search, scoping) share one pool sized for llama-server's parallel slots; report runs,
which hold a slot for tens of minutes, have a pool of their own so they can never starve chat.

Inside the thread the run sets the request's answer language (a context-local override; settings are never touched)
and routes the engine's progress ticks to itself, then records exactly one result or error.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

from web_deploy.api.engine_port import IEngineHost
from web_deploy.api.progress import reporting_to
from web_deploy.api.runs import Run

logger = logging.getLogger(__name__)

# Exception class names (the engine's hierarchy, which this module does not import) and what the browser is told
_KINDS = {
    "EngineBusyError": "busy",
    "ApplianceIngesting": "ingesting",
    "EngineOfflineError": "offline",
    "InvalidPlan": "invalid",
    "ValueError": "invalid",
    "RunNotFound": "not_found",
}


def error_payload(exc: BaseException) -> dict:
    kind = "error"
    for cls in type(exc).__mro__:
        if cls.__name__ in _KINDS:
            kind = _KINDS[cls.__name__]
            break
    message = str(exc) or type(exc).__name__
    return {"kind": kind, "message": message[:2000]}


class Runner:
    def __init__(self, host: IEngineHost, live_workers: int, report_workers: int) -> None:
        self._host = host
        self._live = ThreadPoolExecutor(max_workers=max(1, live_workers), thread_name_prefix="live")
        self._reports = ThreadPoolExecutor(max_workers=max(1, report_workers), thread_name_prefix="report")

    def submit(self, run: Run, work: Callable[[], dict], language: Optional[str], long_running: bool = False) -> None:
        pool = self._reports if long_running else self._live
        pool.submit(self._execute, run, work, language)

    def _execute(self, run: Run, work: Callable[[], dict], language: Optional[str]) -> None:
        try:
            with reporting_to(run):
                if language:
                    with self._host.language_scope(language):
                        result = work()
                else:
                    result = work()
            run.finish(result)
        except Exception as exc:
            logger.warning(f"Run {run.id} ({run.kind}) for {run.principal.firm} failed: {exc}")
            run.fail(error_payload(exc))

    def shutdown(self) -> None:
        self._live.shutdown(wait=False, cancel_futures=True)
        self._reports.shutdown(wait=False, cancel_futures=True)


def call_in_scope(host: IEngineHost, language: Optional[str], work: Callable[[], Any]) -> Any:
    """A short synchronous engine call (a history read) under the request's language, for asyncio.to_thread."""
    if language:
        with host.language_scope(language):
            return work()
    return work()
