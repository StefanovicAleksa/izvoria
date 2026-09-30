# Canvas: desktop_status_bar_v1
# Path: frontend/onyx_desktop/ui/widgets/status_bar.py

"""The appliance's state, always visible. A coloured dot plus what it is currently doing."""

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QWidget

from onyx_desktop.domain.entities import StatusSnapshot
from onyx_desktop.domain.enums import ApplianceStatus
from onyx_desktop.ui import theme

STATUS_LABELS = {
    ApplianceStatus.OFFLINE: "Offline",
    ApplianceStatus.BOOTING: "Starting",
    ApplianceStatus.READY: "Ready",
    ApplianceStatus.WORKING: "Working",
    ApplianceStatus.INGESTING: "Ingesting",
    ApplianceStatus.RESEARCHING: "Researching",
    ApplianceStatus.ERROR: "Error",
}


class StatusBar(QFrame):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("StatusBar")
        self.setFixedHeight(30)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(8)

        self._dot = QLabel("●")
        layout.addWidget(self._dot)

        self._state = QLabel("Offline")
        self._state.setProperty("role", "strong")
        layout.addWidget(self._state)

        self._detail = QLabel("")
        self._detail.setProperty("role", "subtle")
        layout.addWidget(self._detail, stretch=1)

        self._phase = QLabel("")
        self._phase.setProperty("role", "subtle")
        self._phase.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self._phase)

        self.apply(StatusSnapshot(status=ApplianceStatus.OFFLINE, detail="Not started"))

    def apply_theme(self) -> None:
        self.apply(self._snapshot)

    def apply(self, snapshot: StatusSnapshot) -> None:
        self._snapshot = snapshot
        colour = theme.STATUS_COLOURS.get(snapshot.status.value, theme.TEXT_MUTED)
        self._dot.setStyleSheet(f"color: {colour}; font-size: 12px;")
        label = STATUS_LABELS.get(snapshot.status, snapshot.status.value.title())
        self._state.setText(label)
        # "Ready · Ready" says nothing twice
        self._detail.setText("" if snapshot.detail == label else snapshot.detail)

    def set_phase(self, text: str) -> None:
        self._phase.setText(text)
