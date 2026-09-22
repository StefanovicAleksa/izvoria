# Canvas: desktop_ingest_view_v1
# Path: frontend/onyx_desktop/ui/views/ingest_view.py

"""
Point the appliance at a folder and watch it work.

Two honest constraints shape this view:

- **There is no cancellation.** Nothing in the pipeline has a cooperative stop point, and the
  thread sits inside native transcription and LLM calls that ignore signals. So there is no Cancel
  button; there is a disabled one that explains why. A button that pretended to work would be
  worse than none.
- **Progress is two-level and partly blind.** The pipeline runs phase by phase over all files, so
  "file 7 of 23" only means something within the current phase. Several of the longest phases
  (transcription, diarization) log nothing at all, so the bar can sit still for many minutes while
  the box is working hard. The elapsed-silence note exists to say so rather than look hung.
"""

import time
from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from onyx_desktop.domain.entities import IngestionSummary, LogLine, ProgressTick
from onyx_desktop.domain.enums import IngestMode, JobProgressState
from onyx_desktop.ui.widgets.log_pane import LogPane
from onyx_desktop.utils.formatting import duration

_SILENCE_WARN_SECONDS = 90


class IngestView(QWidget):
    ingestRequested = Signal(str, object)   # path, IngestMode

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._phase: Optional[str] = None
        self._phase_done = 0
        self._started_at: Optional[float] = None
        self._last_signal_at = time.monotonic()
        self._build()

        self._silence_timer = QTimer(self)
        self._silence_timer.setInterval(5000)
        self._silence_timer.timeout.connect(self._check_silence)

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(12)

        title = QLabel("Ingest")
        title.setProperty("role", "heading")
        layout.addWidget(title)

        blurb = QLabel(
            "Runs the full pipeline over a folder: transcription, diarization, correction, "
            "visual analysis and indexing. This takes a long time and holds the GPU, so chat "
            "and search are unavailable until it finishes."
        )
        blurb.setProperty("role", "subtle")
        blurb.setWordWrap(True)
        layout.addWidget(blurb)

        picker = QHBoxLayout()
        picker.setSpacing(8)

        self._path = QLineEdit()
        self._path.setPlaceholderText("Absolute path to a folder of media…")
        picker.addWidget(self._path, stretch=1)

        browse = QPushButton("Browse…")
        browse.clicked.connect(self._on_browse)
        picker.addWidget(browse)

        self._mode = QComboBox()
        self._mode.addItem("Video pipeline (full)", IngestMode.VIDEO)
        self._mode.addItem("Audio pipeline (faster)", IngestMode.AUDIO)
        self._mode.setToolTip(
            "The audio pipeline skips visual analysis and extracts no per-chunk clips, so its "
            "citations point at the whole source file."
        )
        picker.addWidget(self._mode)

        self._start = QPushButton("Start ingestion")
        self._start.setProperty("role", "primary")
        self._start.clicked.connect(self._on_start)
        picker.addWidget(self._start)

        self._cancel = QPushButton("Cancel")
        self._cancel.setEnabled(False)
        self._cancel.setToolTip(
            "Ingestion cannot be cancelled. The pipeline has no stop point and runs inside "
            "native transcription and LLM calls that cannot be interrupted safely."
        )
        picker.addWidget(self._cancel)

        layout.addLayout(picker)

        self._phase_label = QLabel("Idle")
        self._phase_label.setStyleSheet("font-weight: 600;")
        layout.addWidget(self._phase_label)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._note = QLabel("")
        self._note.setProperty("role", "subtle")
        self._note.setWordWrap(True)
        layout.addWidget(self._note)

        self._report = _ReportCard()
        self._report.setVisible(False)
        layout.addWidget(self._report)

        log_label = QLabel("Engine output")
        log_label.setProperty("role", "subtle")
        layout.addWidget(log_label)

        self._log = LogPane()
        layout.addWidget(self._log, stretch=1)

    # ---------- public API ----------

    def set_busy(self, busy: bool) -> None:
        self._start.setEnabled(not busy)
        self._path.setEnabled(not busy)
        self._mode.setEnabled(not busy)

    def append_log(self, line: LogLine) -> None:
        self._log.append_line(line)
        self._last_signal_at = time.monotonic()

    def set_phase_from_log(self, banner: str) -> None:
        self._phase_label.setText(banner)

    def apply_tick(self, tick: ProgressTick) -> None:
        """
        One file of one phase. The denominator is unknown from here (the pipeline never announces
        how many files a phase will process), so the bar stays indeterminate and the count is
        reported as a running total instead of a fraction -- an invented denominator would be worse
        than none.
        """
        self._last_signal_at = time.monotonic()

        if tick.phase != self._phase:
            self._phase = tick.phase
            self._phase_done = 0

        if tick.state is JobProgressState.COMPLETED:
            self._phase_done += 1
        elif tick.state is JobProgressState.FAILED:
            self._phase_done += 1
            self._log.append_line(LogLine(
                level="ERROR",
                logger_name="ingest",
                message=f"{tick.phase} failed for {tick.source_id}: {tick.detail or 'no detail'}",
            ))

        pretty = tick.phase.replace("_", " ")
        self._phase_label.setText(f"{pretty} — {self._phase_done} file(s) done")
        self._note.setText("")

    def start_run(self) -> None:
        self._report.setVisible(False)
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)
        self._phase = None
        self._phase_done = 0
        self._started_at = time.monotonic()
        self._last_signal_at = time.monotonic()
        self._phase_label.setText("Starting…")
        self._note.setText("")
        self._silence_timer.start()

    def show_summary(self, summary: IngestionSummary) -> None:
        self._finish()
        self._phase_label.setText("Finished")
        self._report.apply(summary)
        self._report.setVisible(True)

    def show_failure(self, message: str) -> None:
        self._finish()
        self._phase_label.setText("Failed")
        self._note.setText(message)
        self._note.setProperty("role", "banner-danger")
        self._note.style().unpolish(self._note)
        self._note.style().polish(self._note)

    # ---------- internals ----------

    def _finish(self) -> None:
        self._silence_timer.stop()
        self._progress.setVisible(False)

    def _check_silence(self) -> None:
        silent_for = time.monotonic() - self._last_signal_at
        if silent_for < _SILENCE_WARN_SECONDS:
            return
        elapsed = (
            duration(time.monotonic() - self._started_at) if self._started_at else "a while"
        )
        self._note.setText(
            f"Still working — no output for {duration(silent_for)} (running {elapsed}). "
            "Transcription and diarization report nothing while they run; this is expected."
        )

    def _on_browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose a folder to ingest")
        if folder:
            self._path.setText(folder)

    def _on_start(self) -> None:
        path = self._path.text().strip()
        if not path:
            return
        # Rebuilt from the combo's QVariant: a str-derived Enum comes back as a plain str, and an
        # `is` comparison in the gateway would then silently run the video pipeline on audio.
        self.ingestRequested.emit(path, IngestMode(self._mode.currentData()))


class _ReportCard(QFrame):
    """The final IngestionReport, laid out as counters plus any per-file errors."""

    def __init__(self) -> None:
        super().__init__()
        self.setProperty("role", "card")
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(14, 12, 14, 12)
        self._grid.setHorizontalSpacing(24)
        self._grid.setVerticalSpacing(4)

        self._values = {}
        for column, key in enumerate(
            ("scanned", "succeeded", "failed", "vectors", "clips", "elapsed")
        ):
            value = QLabel("0")
            value.setProperty("role", "metric")
            caption = QLabel(key.title())
            caption.setProperty("role", "subtle")
            self._grid.addWidget(value, 0, column, alignment=Qt.AlignmentFlag.AlignLeft)
            self._grid.addWidget(caption, 1, column, alignment=Qt.AlignmentFlag.AlignLeft)
            self._values[key] = value

        self._errors = QLabel("")
        self._errors.setWordWrap(True)
        self._errors.setVisible(False)
        self._grid.addWidget(self._errors, 2, 0, 1, 6)

    def apply(self, summary: IngestionSummary) -> None:
        self._values["scanned"].setText(str(summary.scanned))
        self._values["succeeded"].setText(str(summary.succeeded))
        self._values["failed"].setText(str(summary.failed))
        self._values["vectors"].setText(str(summary.vectors_created))
        self._values["clips"].setText(str(summary.clips_extracted))
        self._values["elapsed"].setText(duration(summary.duration_seconds))

        if summary.errors:
            lines = "\n".join(f"• {key}: {message}" for key, message in summary.errors)
            self._errors.setText(lines)
            self._errors.setProperty("role", "banner-warning")
            self._errors.setVisible(True)
        else:
            self._errors.setVisible(False)
