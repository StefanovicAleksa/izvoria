# Canvas: desktop_log_pane_v1
# Path: frontend/onyx_desktop/ui/widgets/log_pane.py

"""
A scrolling view of the engine's log output.

During a batch ingestion this is often the only sign of life: several of the longest phases log
nothing at all, so the model manager's VRAM narration is what tells an operator the box is still
working rather than hung.
"""

from typing import Optional

from PySide6.QtWidgets import QPlainTextEdit, QWidget

from onyx_desktop.domain.entities import LogLine

_LEVEL_PREFIX = {
    "WARNING": "⚠ ",
    "ERROR": "✗ ",
    "CRITICAL": "✗ ",
}


class LogPane(QPlainTextEdit):
    """Append-only, capped, and auto-scrolling unless the user has scrolled up to read."""

    def __init__(self, max_blocks: int = 3000, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("LogPane")
        self.setReadOnly(True)
        self.setMaximumBlockCount(max_blocks)
        self.setPlaceholderText("Engine output appears here once the appliance is running.")

    def append_line(self, line: LogLine) -> None:
        scrollbar = self.verticalScrollBar()
        # Only follow the tail if the user is already at it, so scrolling back to read a warning
        # is not yanked away by the next log line.
        pinned_to_bottom = scrollbar.value() >= scrollbar.maximum() - 4

        prefix = _LEVEL_PREFIX.get(line.level, "")
        short_name = line.logger_name.split(".")[-1]
        self.appendPlainText(f"{prefix}[{short_name}] {line.message}")

        if pinned_to_bottom:
            scrollbar.setValue(scrollbar.maximum())
