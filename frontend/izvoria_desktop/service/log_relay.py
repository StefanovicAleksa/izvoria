# Canvas: desktop_log_relay_v1
# Path: frontend/izvoria_desktop/service/log_relay.py

"""
Streams the engine's own logging into the UI.

The pipeline narrates itself through module-scoped loggers under the `app.` tree -- phase banners,
per-file ticks, and the model manager's VRAM swaps. None of that reaches a GUI by default, because
the root logger is left at WARNING. Attaching a handler is the cheapest window into a batch run
that otherwise reports nothing for minutes at a time.

Eleven of the batch phases have no logger at all (including transcription and diarization, the two
longest), and one uses bare `print`, so this is a supplement to the JobManager progress ticks
rather than a replacement for them.
"""

import logging
from collections import deque
from typing import Deque, Optional

from PySide6.QtCore import QObject, Signal

from izvoria_desktop.domain.entities import LogLine

# Phase banners look like "Executing RAG Indexing Phase for 4 transcripts..." or
# "========== Starting Izvoria Batch Pipeline for [/path] ==========".
_PHASE_MARKERS = ("Executing ", "Starting Izvoria Batch Pipeline", "Dispatching Phase")


class LogRelay(QObject):
    """
    A `logging.Handler` bridged to Qt.

    `emit()` is called on whichever thread did the logging -- the engine worker, or the model
    manager's background preloader. Emitting a Qt signal across a queued connection is safe from
    any thread; touching a widget from here would not be, which is why this class knows nothing
    about widgets.

    The handler is deliberately not attached in __init__ so tests can build one without mutating
    global logging state.
    """

    lineLogged = Signal(object)   # LogLine
    phaseChanged = Signal(str)

    def __init__(self, capacity: int = 2000, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._buffer: Deque[LogLine] = deque(maxlen=capacity)
        self._handler: Optional[logging.Handler] = None
        self._logger_name = "app"
        self._previous_level: Optional[int] = None

    @property
    def buffered(self) -> tuple:
        """Recent lines, oldest first. Bounded, so a chatty INFO stream cannot grow without end."""
        return tuple(self._buffer)

    def attach(self, logger_name: str = "app", level: int = logging.INFO) -> None:
        """
        Attaches to the engine's logger tree and raises it to INFO. Records propagate up to `app`,
        so one handler covers every phase, feature and the model manager.
        """
        if self._handler is not None:
            return
        self._logger_name = logger_name
        target = logging.getLogger(logger_name)
        self._previous_level = target.level
        target.setLevel(level)

        self._handler = _QtSignalHandler(self)
        self._handler.setLevel(level)
        target.addHandler(self._handler)

    def detach(self) -> None:
        if self._handler is None:
            return
        target = logging.getLogger(self._logger_name)
        target.removeHandler(self._handler)
        if self._previous_level is not None:
            target.setLevel(self._previous_level)
        self._handler = None
        self._previous_level = None

    def _ingest(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:
            # A broken format string in the engine must not take the UI down with it.
            message = str(record.msg)

        line = LogLine(
            level=record.levelname,
            logger_name=record.name,
            message=message,
        )
        self._buffer.append(line)
        self.lineLogged.emit(line)

        if any(marker in message for marker in _PHASE_MARKERS):
            self.phaseChanged.emit(message.strip("= ").strip())


class _QtSignalHandler(logging.Handler):
    """
    The handler half, kept separate from the QObject.

    It does no formatting, no I/O and takes no locks of its own beyond the one `logging` already
    holds: anything slower here would stall the thread doing the actual pipeline work. It also
    swallows every exception, because a logging handler that raises would surface as a pipeline
    failure -- an observation layer must never be able to fail the thing it observes.
    """

    def __init__(self, relay: LogRelay):
        super().__init__()
        self._relay = relay

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._relay._ingest(record)
        except Exception:
            pass
