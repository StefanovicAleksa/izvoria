# Canvas: desktop_system_open_v1
# Path: frontend/izvoria_desktop/ui/system_open.py

"""
Handing a source file to the desktop's own player -- or, on the hosted demo box, to the tester's browser.

One function, used by both the evidence card and the source panel, so the checks that matter --
is there a path at all, is the file still on disk, did the desktop accept it -- cannot drift apart
between the two places a user can click.
"""

import logging
import os
from typing import Optional

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices

from izvoria_desktop.data.source_offer_client import SourceOfferClient
from izvoria_desktop.domain.entities import Evidence
from izvoria_desktop.utils.i18n import tr

logger = logging.getLogger(__name__)


def delivers_to_browser() -> bool:
    """True on the hosted demo box, where sources open in the tester's browser rather than on this desktop."""
    return SourceOfferClient.from_env() is not None


def open_source(evidence: Evidence) -> Optional[str]:
    """
    Opens the evidence's file in whatever the desktop uses for that type. On the hosted box (the agent set
    the offer variables when it launched the app) it goes to the browser instead, where it plays with sound,
    opens at the cited page and can be downloaded; the box's own player would be watched through noVNC.

    Returns None on success, or a sentence explaining why it could not open -- never raises, and
    never fails silently, because a citation that cannot be checked has to say so.
    """
    path = evidence.clip_path
    if not path:
        return (
            tr("The source file for this citation is not available, so there is nothing to open.")
        )

    # A vault row can outlive its file. Saying so beats the desktop quietly doing nothing.
    if not os.path.exists(path):
        logger.warning(f"Cited file is missing from the vault: {path}")
        return tr("This file is no longer available: {name}", name=os.path.basename(path))

    browser = SourceOfferClient.from_env()
    if browser is not None:
        return browser.offer(evidence)

    if not QDesktopServices.openUrl(QUrl.fromLocalFile(path)):
        return tr(
            "The desktop could not open this file — no default application is registered "
            "for this file type."
        )

    logger.info(f"Handed {path} to the system player.")
    return None
