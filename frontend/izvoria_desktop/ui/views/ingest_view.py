# Canvas: desktop_ingest_view_v1
# Path: frontend/izvoria_desktop/ui/views/ingest_view.py

"""
Point the appliance at a folder and watch it work.

Progress is shown as a few general steps (preparing, analyzing, contextualizing, indexing), each with its own
bar -- never the engine's internal jobs or its log. `utils/progress_stages.py` folds jobs into steps.

There is no cancellation: nothing in the pipeline has a safe stop point, so the Cancel button stays disabled
and says so rather than pretending to work.
"""

import time
import uuid
from typing import Dict, List, Optional, Set

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from izvoria_desktop.domain.entities import IngestionSummary, LanguageOption, ProgressTick
from izvoria_desktop.domain.enums import IngestMode, JobProgressState
from izvoria_desktop.ui import translate as tl
from izvoria_desktop.ui.widgets.stage_list import StageList
from izvoria_desktop.utils.error_mapper import file_label, file_problem
from izvoria_desktop.utils.formatting import duration
from izvoria_desktop.utils.i18n import N_, tr, tr_n
from izvoria_desktop.utils.progress_stages import JOBS_PER_STAGE, PREPARING, STAGES_BY_MODE, stage_for

_SILENCE_WARN_SECONDS = 90


class IngestView(QWidget):
    ingestRequested = Signal(str, object, str)   # path, IngestMode, language code

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._run_mode = IngestMode.VIDEO
        self._files: Set[uuid.UUID] = set()
        self._done_in_stage: Dict[str, int] = {}
        self._started_at: Optional[float] = None
        self._last_signal_at = time.monotonic()
        self._build()

        self._silence_timer = QTimer(self)
        self._silence_timer.setInterval(5000)
        self._silence_timer.timeout.connect(self._check_silence)

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)

        layout.addWidget(tl.label("Ingest", role="heading"))

        blurb = tl.label(
            "Adds a folder of material to the archive so it can be searched and cited. Large batches take a "
            "while, and chat and search are unavailable until it finishes.",
            role="subtle",
        )
        blurb.setWordWrap(True)
        layout.addWidget(blurb)

        picker = QHBoxLayout()
        picker.setSpacing(8)

        self._path = QLineEdit()
        tl.placeholder(self._path, "Folder to add…")
        picker.addWidget(self._path, stretch=1)

        browse = QPushButton()
        tl.text(browse, "Browse…")
        browse.clicked.connect(self._on_browse)
        picker.addWidget(browse)

        self._mode = QComboBox()
        for mode in (IngestMode.VIDEO, IngestMode.AUDIO, IngestMode.DOCUMENT, IngestMode.IMAGE):
            self._mode.addItem("", mode)
        tl.items(self._mode, [
            "Video", "Audio",
            lambda: tr("Documents & books ({types})", types="PDF, DOCX, EPUB, TXT, MD, HTML, RTF"),
            lambda: tr("Images ({types})", types="JPG, PNG, TIFF, WebP"),
        ])
        tl.tooltip(self._mode, "What the folder holds. Recordings are cited by time, documents by page, line or "
                               "section, and images as a whole.")
        # Sized for the short names; the list still shows each in full, so the folder field keeps its room
        self._mode.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self._mode.setMinimumContentsLength(16)
        picker.addWidget(self._mode)

        # Filled from the appliance's language packs by set_languages()
        self._language = QComboBox()
        tl.tooltip(self._language, "The language spoken or written in these files. A recording in a different "
                                   "language is skipped rather than processed in the wrong one.")
        picker.addWidget(tl.label("Language of these files:"))
        picker.addWidget(self._language)

        self._start = QPushButton()
        tl.text(self._start, "Start ingestion")
        self._start.setProperty("role", "primary")
        self._start.clicked.connect(self._on_start)
        picker.addWidget(self._start)

        self._cancel = QPushButton()
        tl.text(self._cancel, "Cancel")
        self._cancel.setEnabled(False)
        tl.tooltip(self._cancel, "A batch cannot be cancelled once it has started.")
        picker.addWidget(self._cancel)

        layout.addLayout(picker)

        self._phase_label = tl.label("Choose a folder and start. Progress appears here.", role="subtle")
        layout.addWidget(self._phase_label)

        self._stages = StageList()
        self._stages.setVisible(False)
        layout.addWidget(self._stages)

        # Shown for the whole run: the longest steps report only as each file finishes, so a bar can hold still
        # for a long time, and the natural reaction to that -- closing the app -- is the one that does harm.
        self._patience = tl.label(
            "Some steps take considerably longer than others, especially with long recordings or large "
            "documents, and a bar may stay in place for several minutes while work continues in the "
            "background. This is expected: there is no need to restart the appliance or close the app. "
            "Progress will move on as each file is completed.",
            role="notice",
        )
        self._patience.setWordWrap(True)
        self._patience.setVisible(False)
        layout.addWidget(self._patience)

        self._note = QLabel("")
        self._note.setProperty("role", "subtle")
        self._note.setWordWrap(True)
        layout.addWidget(self._note)

        self._report = _ReportCard()
        self._report.setVisible(False)
        layout.addWidget(self._report)

        layout.addStretch(1)

    # ---------- public API ----------

    def set_busy(self, busy: bool) -> None:
        self._start.setEnabled(not busy)
        self._path.setEnabled(not busy)
        self._mode.setEnabled(not busy)
        self._language.setEnabled(not busy)

    def set_languages(self, options: List[LanguageOption], selected: str) -> None:
        self._language.clear()
        for option in options:
            self._language.addItem(option.name, option.code)
        self.select_language(selected)

    def select_language(self, code: str) -> None:
        """Defaults the next batch to `code` (the active language); the user can still pick another."""
        index = self._language.findData(code)
        if index >= 0:
            self._language.setCurrentIndex(index)

    def apply_tick(self, tick: ProgressTick) -> None:
        """Moves the step the tick's job belongs to. A job outside every step is not shown at all."""
        stage = stage_for(tick.phase)
        if stage is None or stage.key not in self._stages.keys:
            return
        self._last_signal_at = time.monotonic()
        tl.text(self._note, tl.plain(""))
        if tick.source_id is not None:
            self._files.add(tick.source_id)

        if tick.state in (JobProgressState.COMPLETED, JobProgressState.FAILED):
            self._done_in_stage[stage.key] = self._done_in_stage.get(stage.key, 0) + 1

        self._stages.advance(stage.key, self._fraction(stage.key))
        self._show_position(stage.key)

    def start_run(self) -> None:
        self._run_mode = IngestMode(self._mode.currentData())
        self._files.clear()
        self._done_in_stage.clear()
        self._report.setVisible(False)
        self._stages.set_stages(STAGES_BY_MODE[self._run_mode])
        self._stages.setVisible(True)
        self._patience.setVisible(True)
        self._stages.advance(PREPARING.key, None)
        self._started_at = time.monotonic()
        self._last_signal_at = time.monotonic()
        self._show_position(PREPARING.key)
        self._reset_note()
        self._silence_timer.start()

    def show_summary(self, summary: IngestionSummary) -> None:
        self._finish()
        self._stages.finish()
        tl.text(self._phase_label, "Finished")
        self._report.apply(summary)
        self._report.setVisible(True)

    def show_failure(self, message: str) -> None:
        self._finish()
        self._stages.fail()
        tl.text(self._phase_label, "Stopped")
        tl.text(self._note, tl.plain(message))   # already in the display language
        self._note.setProperty("role", "banner-danger")
        self._note.style().unpolish(self._note)
        self._note.style().polish(self._note)

    # ---------- internals ----------

    def _finish(self) -> None:
        self._silence_timer.stop()
        self._patience.setVisible(False)

    def _fraction(self, key: str) -> Optional[float]:
        """How far through its step the batch is, estimated from files done; None while the batch size is unknown."""
        if key == PREPARING.key or not self._files:
            return None
        expected = len(self._files) * JOBS_PER_STAGE.get((self._run_mode, key), 1)
        return self._done_in_stage.get(key, 0) / expected

    def _show_position(self, key: str) -> None:
        keys = self._stages.keys
        files = len(self._files)
        position = keys.index(key) + 1

        def line() -> str:
            step = tr("Step {current} of {total}", current=position, total=len(keys))
            return f"{step} · {tr_n(files, '{n} file', '{n} files')}" if files else step

        tl.text(self._phase_label, line)

    def _reset_note(self) -> None:
        tl.text(self._note, tl.plain(""))
        self._note.setProperty("role", "subtle")
        self._note.style().unpolish(self._note)
        self._note.style().polish(self._note)

    def _check_silence(self) -> None:
        silent_for = time.monotonic() - self._last_signal_at
        if silent_for < _SILENCE_WARN_SECONDS:
            return
        started = self._started_at or time.monotonic()
        tl.text(self._note, lambda: tr(
            "Still working — {elapsed} so far. The current step is taking a while, which is normal for larger "
            "files.", elapsed=duration(time.monotonic() - started),
        ))

    def _on_browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, tr("Choose a folder to add"))
        if folder:
            self._path.setText(folder)

    def _on_start(self) -> None:
        path = self._path.text().strip()
        if not path:
            return
        # Rebuilt from the combo's QVariant: a str-derived Enum comes back as a plain str, and an
        # `is` comparison in the gateway would then silently run the video pipeline on audio.
        self.ingestRequested.emit(path, IngestMode(self._mode.currentData()), str(self._language.currentData()))


class _ReportCard(QFrame):
    """How the batch went: how many files, how many made it in, and which did not and why, in plain words."""

    def __init__(self) -> None:
        super().__init__()
        self.setProperty("role", "card")
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(18, 14, 18, 14)
        self._grid.setHorizontalSpacing(32)
        self._grid.setVerticalSpacing(4)

        self._values = {}
        for column, (key, caption_text) in enumerate(
            (("scanned", N_("Files")), ("succeeded", N_("Added")), ("failed", N_("Not added")),
             ("elapsed", N_("Time")))
        ):
            value = QLabel("0")
            value.setProperty("role", "metric")
            caption = tl.label(caption_text, role="subtle")
            self._grid.addWidget(value, 0, column, alignment=Qt.AlignmentFlag.AlignLeft)
            self._grid.addWidget(caption, 1, column, alignment=Qt.AlignmentFlag.AlignLeft)
            self._values[key] = value
        self._grid.setColumnStretch(4, 1)

        self._errors = QLabel("")
        self._errors.setWordWrap(True)
        self._errors.setVisible(False)
        self._grid.addWidget(self._errors, 2, 0, 1, 5)

    def apply(self, summary: IngestionSummary) -> None:
        self._values["scanned"].setText(str(summary.scanned))
        self._values["succeeded"].setText(str(summary.succeeded))
        self._values["failed"].setText(str(summary.failed))
        tl.text(self._values["elapsed"], lambda: duration(summary.duration_seconds))

        if summary.errors:
            tl.text(self._errors, lambda: "\n".join(
                f"• {file_label(key)}: {file_problem(message)}" for key, message in summary.errors))
            self._errors.setProperty("role", "banner-warning")
            self._errors.setVisible(True)
        else:
            self._errors.setVisible(False)
