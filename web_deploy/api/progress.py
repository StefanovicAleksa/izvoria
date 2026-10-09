"""
Routes the engine's progress ticks to the run that caused them.

A firm's engine is shared by all of that firm's concurrent requests, and its JobManager has one listener. That
listener reads which run the current thread is working for from a ContextVar set by the runner, so ticks from two
users' simultaneous chats go to their own runs. Work handed to another thread inside the engine (the scrubber's
parallel judges) runs in a copied context, so it still reaches the right run.
"""

import contextvars
import uuid
from contextlib import contextmanager
from typing import Iterator, Optional

from web_deploy.api.runs import Run

_current_run: contextvars.ContextVar[Optional[Run]] = contextvars.ContextVar("web_deploy_run", default=None)


@contextmanager
def reporting_to(run: Run) -> Iterator[Run]:
    token = _current_run.set(run)
    try:
        yield run
    finally:
        _current_run.reset(token)


def job_listener(phase: str, state: str, source_id: Optional[uuid.UUID], detail: Optional[str] = None) -> None:
    """The JobManager progress listener given to every firm's engine. Must never raise."""
    run = _current_run.get()
    if run is None:
        return
    try:
        run.emit("progress", {
            "phase": str(phase),
            "state": str(state),
            "source_id": str(source_id) if source_id else None,
            "detail": detail,
        })
    except Exception:
        pass
