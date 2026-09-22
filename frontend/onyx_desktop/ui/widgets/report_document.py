# Canvas: desktop_report_document_v1
# Path: frontend/onyx_desktop/ui/widgets/report_document.py

"""
The rendered report, with citations you can actually click.

Worth being precise about what is new here: the chat view's `[Doc N]` chips only look clickable --
`MessageBubble._body()` renders them as a coloured `<span>`, and you verify a chat claim by clicking
the evidence *card* underneath. A report is pages long and has no cards, so the citation itself has
to be the control.

`QTextBrowser` is the app's first rich-text widget. It is the right one because `anchorClicked` plus
`setOpenLinks(False)` pairs directly with the `onyx://doc/N` scheme the engine's markdown renderer
already emits in link mode -- that mode exists for exactly this.
"""

import logging
import re
from typing import Optional

from PySide6.QtCore import QUrl, Signal
from PySide6.QtWidgets import QTextBrowser, QWidget

from onyx_desktop.domain.entities import Evidence, ReportDocument
from onyx_desktop.ui import theme

logger = logging.getLogger(__name__)

_DOC_URL = re.compile(r"^onyx://doc/(\d+)$")


class ReportDocumentView(QTextBrowser):
    """Renders a report's Markdown and turns a citation click into an Evidence signal."""

    citationClicked = Signal(object)   # Evidence
    citationMissing = Signal(int)      # doc number whose evidence is gone

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._report: Optional[ReportDocument] = None

        # Never let Qt hand a URL to a system browser: onyx:// means nothing outside this app, and a
        # report's links are internal navigation, not the web.
        self.setOpenLinks(False)
        self.setOpenExternalLinks(False)
        self.anchorClicked.connect(self._on_anchor)

        self.setStyleSheet(
            f"QTextBrowser {{ background-color: {theme.SURFACE}; border: 1px solid {theme.BORDER};"
            f" border-radius: 8px; padding: 16px; font-size: 14px; }}"
            f"QTextBrowser a {{ color: {theme.CITATION}; text-decoration: none; font-weight: 600; }}"
        )

    # ---------- public API ----------

    def show_report(self, report: ReportDocument) -> None:
        self._report = report
        self.setMarkdown(report.markdown)
        self.verticalScrollBar().setValue(0)

    def clear_report(self) -> None:
        self._report = None
        self.setMarkdown("")

    # ---------- events ----------

    def _on_anchor(self, url: QUrl) -> None:
        """
        Resolves `onyx://doc/N` against the report's evidence bank.

        A number with no evidence is reported rather than ignored: it means the excerpt behind a
        citation is gone (a re-ingest deleted its chunk), and a citation that cannot be checked has to
        say so -- the same rule `system_open` follows for a missing media file.
        """
        match = _DOC_URL.match(url.toString())
        if match is None or self._report is None:
            return

        doc_number = int(match.group(1))
        evidence: Optional[Evidence] = self._report.evidence_for(doc_number)
        if evidence is None:
            logger.info(f"Citation [Doc {doc_number}] has no evidence in this report's bank.")
            self.citationMissing.emit(doc_number)
            return

        self.citationClicked.emit(evidence)
