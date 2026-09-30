# Canvas: desktop_appearance_v1
# Path: frontend/onyx_desktop/ui/appearance.py

"""Remembers the light/dark choice between runs. A per-user preference, so it lives in QSettings, not the appliance database."""

from typing import Optional

from PySide6.QtCore import QSettings

from onyx_desktop.ui import theme

_KEY = "appearance/mode"


def saved_mode() -> Optional[str]:
    """The mode the user last picked in the app, or None if they never did (or it is no longer a mode)."""
    value = QSettings("Onyx", "Onyx Desktop").value(_KEY)
    return value if value in theme.MODES else None


def save_mode(mode: str) -> None:
    QSettings("Onyx", "Onyx Desktop").setValue(_KEY, mode)
