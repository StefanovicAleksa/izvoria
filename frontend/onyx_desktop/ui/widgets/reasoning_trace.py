# Canvas: desktop_reasoning_trace_v1
# Path: frontend/onyx_desktop/ui/widgets/reasoning_trace.py

"""
The Reasoner's decision history, collapsed by default.

Worth showing because it is the only view into *why* a multi-pass answer searched what it did --
and, when the reasoner asks for clarification instead of answering, the only explanation of what
it was missing.
"""

from typing import Optional, Sequence

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from onyx_desktop.domain.entities import ReasoningStep
from onyx_desktop.ui import theme


class ReasoningTrace(QWidget):
    """A compact, expandable list of reasoner iterations."""

    def __init__(self, steps: Sequence[ReasoningStep], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._steps = tuple(steps)
        self._expanded = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._toggle = QLabel()
        self._toggle.setOpenExternalLinks(False)
        self._toggle.linkActivated.connect(self._on_toggle)
        layout.addWidget(self._toggle)

        self._detail = QLabel()
        self._detail.setWordWrap(True)
        self._detail.setProperty("role", "subtle")
        self._detail.setVisible(False)
        self._detail.setStyleSheet(
            f"border-left: 2px solid {theme.BORDER}; padding-left: 10px;"
        )
        layout.addWidget(self._detail)

        self._render()

    def _on_toggle(self, _link: str) -> None:
        self._expanded = not self._expanded
        self._render()

    def _render(self) -> None:
        arrow = "▾" if self._expanded else "▸"
        count = len(self._steps)
        self._toggle.setText(
            f'<a href="#" style="color:{theme.TEXT_MUTED}; text-decoration:none;">'
            f"{arrow} reasoning trace ({count} iteration{'s' if count != 1 else ''})</a>"
        )
        self._detail.setVisible(self._expanded)
        if self._expanded:
            self._detail.setText("<br><br>".join(self._format(s) for s in self._steps))

    @staticmethod
    def _format(step: ReasoningStep) -> str:
        lines = [
            f"<b>Iteration {step.iteration} — {step.decision}</b>",
            step.interim_analysis.replace("<", "&lt;"),
        ]
        if step.refined_query:
            lines.append(f"→ searched: <i>{step.refined_query}</i>")
        if step.chunks_added:
            lines.append(f"+{step.chunks_added} chunk(s) added to the evidence pool")
        return "<br>".join(lines)
