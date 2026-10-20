# Path: frontend/izvoria_desktop/service/link_checker.py

"""
Reads what YouTube links hold, on a thread of its own.

Not an engine call: it needs no model, no database and not even a started appliance, only the network, and a long
playlist takes seconds. On the engine worker it would wait behind an ingestion; on the GUI thread it would freeze
the window.
"""

import logging

from PySide6.QtCore import QObject, Signal, Slot

from izvoria_desktop.domain.interfaces import IYouTubeLinks
from izvoria_desktop.exceptions.base import DesktopError
from izvoria_desktop.utils.i18n import tr

logger = logging.getLogger(__name__)


class LinkChecker(QObject):
    ready = Signal(object)    # LinkCheck
    failed = Signal(str)      # user message

    def __init__(self, links: IYouTubeLinks):
        super().__init__()
        self._links = links

    @Slot(object)
    def run(self, links: list) -> None:
        try:
            self.ready.emit(self._links.check(list(links)))
        except Exception as exc:
            logger.error(f"Checking YouTube links failed: {exc}", exc_info=exc)
            message = exc.user_message if isinstance(exc, DesktopError) else tr(
                "The links could not be checked. Check the internet connection and try again.")
            self.failed.emit(message)
