"""
A long engine call as something a browser can follow: a run.

Every call that takes more than a moment (a chat turn, a search, scoping or writing a report) is started as a run
and returns its id at once. The call works on a worker thread and appends events: `progress` ticks from the
engine's JobManager, then exactly one `result` or `error`. The browser reads them as Server-Sent Events, and a
reconnect resumes from its Last-Event-ID, so a dropped connection mid-report loses nothing. A run belongs to the
user who started it: nobody else can read it.

Runs live in memory for RUN_TTL_SECONDS after they finish. The work they did is in the firm's database either way
(conversation turns, search history, the report), so an API restart costs only the live view of runs in flight.
"""

import itertools
import json
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from web_deploy.common.signing import Principal

RUN_TTL_SECONDS = 30 * 60
MAX_EVENTS_PER_RUN = 5000
# A progress tick per file per phase is plenty for a bar; a report fans out hundreds of rerank jobs
MAX_PROGRESS_EVENTS = 2000


class RunNotFound(LookupError):
    pass


@dataclass
class Run:
    id: str
    principal: Principal
    kind: str
    created_at: float
    events: List[Tuple[int, str, dict]] = field(default_factory=list)
    finished_at: Optional[float] = None
    # Identifies what the run is about (a report id), so the UI can reattach to it after a reload
    subject: Optional[str] = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _seq: Any = field(default_factory=lambda: itertools.count(1), repr=False)
    _progress_count: int = 0

    @property
    def done(self) -> bool:
        return self.finished_at is not None

    def emit(self, event: str, data: dict) -> None:
        with self._lock:
            if self.done:
                return
            if event == "progress":
                self._progress_count += 1
                if self._progress_count > MAX_PROGRESS_EVENTS:
                    return
            if len(self.events) >= MAX_EVENTS_PER_RUN:
                return
            self.events.append((next(self._seq), event, data))

    def finish(self, result: dict) -> None:
        self._end("result", result)

    def fail(self, error: dict) -> None:
        self._end("error", error)

    def _end(self, event: str, data: dict) -> None:
        with self._lock:
            if self.done:
                return
            self.events.append((next(self._seq), event, data))
            self.finished_at = time.monotonic()

    def after(self, seq: int) -> List[Tuple[int, str, dict]]:
        with self._lock:
            return [e for e in self.events if e[0] > seq]

    def public(self) -> dict:
        return {"id": self.id, "kind": self.kind, "done": self.done, "subject": self.subject}


class RunStore:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._runs: Dict[str, Run] = {}
        self._lock = threading.Lock()
        self._clock = clock

    def create(self, principal: Principal, kind: str, subject: Optional[str] = None) -> Run:
        run = Run(id=secrets.token_urlsafe(16), principal=principal, kind=kind, created_at=self._clock(),
                  subject=subject)
        with self._lock:
            self._purge()
            self._runs[run.id] = run
        return run

    def get(self, run_id: str, principal: Principal) -> Run:
        """The run, if it is this user's. Someone else's run is indistinguishable from one that never existed."""
        with self._lock:
            self._purge()
            run = self._runs.get(run_id)
        if run is None or run.principal.firm != principal.firm or run.principal.user_id != principal.user_id:
            raise RunNotFound(run_id)
        return run

    def active(self, principal: Principal) -> List[Run]:
        with self._lock:
            return [r for r in self._runs.values()
                    if not r.done and r.principal.firm == principal.firm and r.principal.user_id == principal.user_id]

    def _purge(self) -> None:
        cutoff = self._clock() - RUN_TTL_SECONDS
        for run_id in [i for i, r in self._runs.items() if r.finished_at is not None and r.finished_at < cutoff]:
            del self._runs[run_id]


def sse_frame(seq: int, event: str, data: dict) -> str:
    return f"id: {seq}\nevent: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def resume_point(last_event_id: Optional[str]) -> int:
    return int(last_event_id) if last_event_id and last_event_id.isdigit() else 0
