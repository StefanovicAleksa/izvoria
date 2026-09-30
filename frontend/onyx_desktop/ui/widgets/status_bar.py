# Canvas: desktop_status_bar_v1
# Path: frontend/onyx_desktop/ui/widgets/status_bar.py

"""The appliance's state, always visible. A coloured dot plus what it is currently doing."""

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QWidget

from onyx_desktop.domain.entities import StatusSnapshot
from onyx_desktop.domain.enums import ApplianceStatus
from onyx_desktop.ui import theme
from onyx_desktop.ui import translate as tl
from onyx_desktop.utils.i18n import N_, tr

# English keys; shown through tr()
STATUS_LABELS = {
    ApplianceStatus.OFFLINE: N_("Offline"),
    ApplianceStatus.BOOTING: N_("Starting"),
    ApplianceStatus.READY: N_("Ready"),
    ApplianceStatus.WORKING: N_("Working"),
    ApplianceStatus.INGESTING: N_("Ingesting"),
    ApplianceStatus.RESEARCHING: N_("Researching"),
    ApplianceStatus.ERROR: N_("Error"),
}


def status_label(status: ApplianceStatus) -> str:
    return tr(STATUS_LABELS.get(status, status.value.title()))


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

        self._state = QLabel()
        self._state.setProperty("role", "strong")
        layout.addWidget(self._state)

        self._detail = QLabel("")
        self._detail.setProperty("role", "subtle")
        layout.addWidget(self._detail, stretch=1)

        self._phase = QLabel("")
        self._phase.setProperty("role", "subtle")
        self._phase.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self._phase)

        self.apply(StatusSnapshot(status=ApplianceStatus.OFFLINE, detail=N_("Not started")))

    def apply_theme(self) -> None:
        self.apply(self._snapshot)

    def apply(self, snapshot: StatusSnapshot) -> None:
        self._snapshot = snapshot
        colour = theme.STATUS_COLOURS.get(snapshot.status.value, theme.TEXT_MUTED)
        self._dot.setStyleSheet(f"color: {colour}; font-size: 12px;")
        tl.text(self._state, lambda: status_label(snapshot.status))
        # Details are English keys from the bridge, or messages already translated; tr() leaves the latter as is.
        # "Ready · Ready" says nothing twice.
        tl.text(self._detail, lambda: "" if snapshot.detail in ("", STATUS_LABELS.get(snapshot.status))
                else tr(snapshot.detail))

    def set_phase(self, text: str) -> None:
        self._phase.setText(text)
