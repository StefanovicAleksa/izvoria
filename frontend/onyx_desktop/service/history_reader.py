# Path: frontend/onyx_desktop/service/history_reader.py

"""
Runs history calls on a thread of their own.

The engine worker runs one call at a time, and an ingestion or a report holds it for hours. History
is a database read that needs no model, so waiting behind the GPU would make it useless exactly when
it matters (picking up where you left off while a long job runs). The history methods are the only
gateway calls allowed here; IEngineGateway documents why they are safe beside the worker.
"""

import logging

from PySide6.QtCore import QObject, Signal, Slot

from onyx_desktop.domain.interfaces import IEngineGateway
from onyx_desktop.exceptions.base import DesktopError
from onyx_desktop.utils.error_mapper import map_engine_error

logger = logging.getLogger(__name__)

# Gateway method -> what to call the failure. Anything else is refused: this thread must never reach a
# method that loads a model.
HISTORY_CALLS = {
    "list_conversations": "Listing conversations",
    "load_conversation": "Loading a conversation",
    "delete_conversation": "Deleting a conversation",
    "list_searches": "Listing past searches",
    "load_search": "Loading a past search",
    "delete_search": "Deleting a past search",
    "list_reports": "Listing reports",
    "load_report": "Loading a report",
    "load_report_plan": "Reopening a research plan",
}


class HistoryReader(QObject):
    ready = Signal(int, str, object)    # request_id, method, payload
    failed = Signal(int, str, str)      # request_id, method, user_message

    def __init__(self, gateway: IEngineGateway):
        super().__init__()
        self._gateway = gateway

    @Slot(int, str, object)
    def run(self, request_id: int, method: str, args: tuple) -> None:
        context = HISTORY_CALLS.get(method)
        if context is None:
            raise ValueError(f"'{method}' is not a history call.")
        try:
            payload = getattr(self._gateway, method)(*args)
            self.ready.emit(request_id, method, payload)
        except Exception as exc:
            error = exc if isinstance(exc, DesktopError) else map_engine_error(exc, context=context)
            logger.error(f"{context} failed: {error.user_message}", exc_info=error.original_exception)
            self.failed.emit(request_id, method, error.user_message)
