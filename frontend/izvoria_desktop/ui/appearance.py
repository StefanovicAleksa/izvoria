# Canvas: desktop_appearance_v1
# Path: frontend/izvoria_desktop/ui/appearance.py

"""
Remembers the light/dark choice and the display language between runs. Per-user preferences, so they live in
QSettings, not the appliance database.
"""

from typing import Optional

from PySide6.QtCore import QSettings

from izvoria_desktop.ui import theme
from izvoria_desktop.utils import i18n

# QSettings keeps the app's pre-rename "Onyx" names, so a user's saved theme and display language survive.
_KEY = "appearance/mode"
_DISPLAY_LANGUAGE_KEY = "appearance/display_language"


def saved_mode() -> Optional[str]:
    """The mode the user last picked in the app, or None if they never did (or it is no longer a mode)."""
    value = QSettings("Onyx", "Onyx Desktop").value(_KEY)
    return value if value in theme.MODES else None


def save_mode(mode: str) -> None:
    QSettings("Onyx", "Onyx Desktop").setValue(_KEY, mode)


def saved_display_language() -> Optional[str]:
    """The display language last picked in the app, or None (never picked, or its catalog has since gone)."""
    value = QSettings("Onyx", "Onyx Desktop").value(_DISPLAY_LANGUAGE_KEY)
    return value if value in {language.code for language in i18n.available()} else None


def save_display_language(code: str) -> None:
    QSettings("Onyx", "Onyx Desktop").setValue(_DISPLAY_LANGUAGE_KEY, code)
