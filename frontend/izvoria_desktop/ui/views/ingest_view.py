# Canvas: desktop_ingest_view_v1
# Path: frontend/izvoria_desktop/ui/views/ingest_view.py

"""
Point the appliance at a folder and watch it work.

Progress is shown as a few general steps (preparing, analyzing, contextualizing, indexing), each with its own
bar -- never the engine's internal jobs or its log. `utils/progress_stages.py` folds jobs into steps.

There is no cancellation: nothing in the pipeline has a safe stop point, so the Cancel button stays disabled
and says so rather than pretending to work.

Where yt-dlp is installed, a Video batch can also come from YouTube links (data/youtube_links.py): "Check links"
lists the videos they hold, the user unticks any they don't want, and the batch downloads them first.
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
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from izvoria_desktop.domain.entities import IngestionSummary, LanguageOption, LinkCheck, ProgressTick, YouTubeVideo
from izvoria_desktop.domain.enums import IngestMode, JobProgressState
from izvoria_desktop.ui import translate as tl
from izvoria_desktop.ui.widgets.stage_list import StageList
from izvoria_desktop.utils.error_mapper import file_label, file_problem
from izvoria_desktop.utils.formatting import duration
from izvoria_desktop.utils.i18n import N_, tr, tr_n
from izvoria_desktop.utils.progress_stages import DOWNLOADING, JOBS_PER_STAGE, PREPARING, stage_for, stages_for

_SILENCE_WARN_SECONDS = 90


class IngestView(QWidget):
    ingestRequested = Signal(str, object, str)   # path, IngestMode, language code
    linksCheckRequested = Signal(object)         # List[str], one per link typed
    linkIngestRequested = Signal(object, str)    # List[YouTubeVideo], language code

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._run_mode = IngestMode.VIDEO
        self._links_available = False
        self._busy = False
        self._checking = False
        self._found: List[YouTubeVideo] = []
        self._run_videos = 0   # videos a link batch downloads; 0 for a folder
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

        self._links = self._build_links()
        self._links.setVisible(False)
        layout.addWidget(self._links)
        self._mode.currentIndexChanged.connect(self._show_links_if_video)

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

    def _build_links(self) -> QFrame:
        card = QFrame()
        card.setProperty("role", "card")
        box = QVBoxLayout(card)
        box.setContentsMargins(18, 14, 18, 14)
        box.setSpacing(8)

        box.addWidget(tl.label("Or add videos from YouTube", role="subtle"))

        self._link_text = QPlainTextEdit()
        tl.placeholder(self._link_text, "Paste YouTube links, one per line: single videos or whole playlists")
        self._link_text.setMaximumHeight(72)
        self._link_text.textChanged.connect(self._refresh_link_buttons)
        box.addWidget(self._link_text)

        row = QHBoxLayout()
        row.setSpacing(8)
        self._check_links = QPushButton()
        tl.text(self._check_links, "Check links")
        tl.tooltip(self._check_links, "Lists the videos the links hold without downloading anything.")
        self._check_links.clicked.connect(self._on_check_links)
        row.addWidget(self._check_links)
        self._links_note = QLabel("")
        self._links_note.setProperty("role", "subtle")
        self._links_note.setWordWrap(True)
        row.addWidget(self._links_note, stretch=1)
        box.addLayout(row)

        self._video_list = QListWidget()
        self._video_list.setVisible(False)
        self._video_list.setMaximumHeight(220)
        self._video_list.itemChanged.connect(lambda _item: self._refresh_link_buttons())
        box.addWidget(self._video_list)
        tl.on_retranslate(self, "video_list", self._retitle_videos)

        self._links_problems = QLabel("")
        self._links_problems.setWordWrap(True)
        self._links_problems.setProperty("role", "banner-warning")
        self._links_problems.setVisible(False)
        box.addWidget(self._links_problems)

        add_row = QHBoxLayout()
        add_row.addStretch(1)
        self._add_videos = QPushButton()
        tl.text(self._add_videos, lambda: tr_n(self._picked_count(), "Add {n} video", "Add {n} videos"))
        self._add_videos.setProperty("role", "primary")
        self._add_videos.setVisible(False)
        self._add_videos.clicked.connect(self._on_add_videos)
        add_row.addWidget(self._add_videos)
        box.addLayout(add_row)

        self._refresh_link_buttons()
        return card

    # ---------- public API ----------

    def set_links_available(self, available: bool) -> None:
        """Whether this machine can take YouTube links (yt-dlp is installed). Shown for Video batches only."""
        self._links_available = available
        self._show_links_if_video()

    def show_link_check(self, check: LinkCheck) -> None:
        self._checking = False
        self._found = list(check.videos)
        self._video_list.blockSignals(True)
        self._video_list.clear()
        for video in self._found:
            item = QListWidgetItem()
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self._video_list.addItem(item)
        self._video_list.blockSignals(False)
        self._retitle_videos()
        self._video_list.setVisible(bool(self._found))
        self._add_videos.setVisible(bool(self._found))

        lines = [f"• {link}: {reason}" for link, reason in check.problems]
        if check.playlist_hints:
            lines.append(tr("• A link to a video inside a playlist adds only that video. To add the whole "
                            "playlist, paste its playlist link."))
        if check.capped:
            lines.append(tr("• Only the first {n} videos were taken: one batch holds at most that many.",
                            n=len(self._found)))
        self._links_problems.setText("\n".join(lines))   # YouTube's own reasons are not translated
        self._links_problems.setVisible(bool(lines))
        if self._found:
            tl.text(self._links_note, lambda: tr_n(len(self._found), "{n} video found.", "{n} videos found."))
        else:
            tl.text(self._links_note, "No videos that can be downloaded were found.")
        self._refresh_link_buttons()

    def show_link_check_failure(self, message: str) -> None:
        self._checking = False
        tl.text(self._links_note, tl.plain(message))   # already in the display language
        self._refresh_link_buttons()

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        self._start.setEnabled(not busy)
        self._path.setEnabled(not busy)
        self._mode.setEnabled(not busy)
        self._language.setEnabled(not busy)
        self._link_text.setEnabled(not busy)
        self._video_list.setEnabled(not busy)
        self._refresh_link_buttons()

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

    def start_run(self, videos: int = 0) -> None:
        """`videos` > 0 for a batch from YouTube links: it downloads that many videos first."""
        self._run_mode = IngestMode(self._mode.currentData()) if not videos else IngestMode.VIDEO
        self._run_videos = videos
        self._files.clear()
        self._done_in_stage.clear()
        self._report.setVisible(False)
        self._stages.set_stages(stages_for(self._run_mode, from_links=bool(videos)))
        self._stages.setVisible(True)
        self._patience.setVisible(True)
        first = DOWNLOADING if videos else PREPARING
        self._stages.advance(first.key, 0.0 if videos else None)
        self._started_at = time.monotonic()
        self._last_signal_at = time.monotonic()
        self._show_position(first.key)
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
        if key == DOWNLOADING.key:
            return self._done_in_stage.get(key, 0) / max(1, self._run_videos)
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

    def _show_links_if_video(self, *_args) -> None:
        self._links.setVisible(self._links_available and IngestMode(self._mode.currentData()) is IngestMode.VIDEO)

    def _typed_links(self) -> List[str]:
        return [line.strip() for line in self._link_text.toPlainText().splitlines() if line.strip()]

    def _picked(self) -> List[YouTubeVideo]:
        return [video for row, video in enumerate(self._found)
                if self._video_list.item(row).checkState() == Qt.CheckState.Checked]

    def _picked_count(self) -> int:
        return len(self._picked()) if self._video_list.count() == len(self._found) else 0

    def _retitle_videos(self) -> None:
        for row, video in enumerate(self._found):
            item = self._video_list.item(row)
            if item is not None:
                item.setText(f"{video.title} · {duration(video.duration)}" if video.duration else video.title)

    def _refresh_link_buttons(self) -> None:
        self._check_links.setEnabled(not self._busy and not self._checking and bool(self._typed_links()))
        self._add_videos.setEnabled(not self._busy and self._picked_count() > 0)
        tl.retranslate_widget(self._add_videos)

    def _on_check_links(self) -> None:
        links = self._typed_links()
        if not links:
            return
        self._checking = True
        tl.text(self._links_note, "Checking the links…")
        self._refresh_link_buttons()
        self.linksCheckRequested.emit(links)

    def _on_add_videos(self) -> None:
        picked = self._picked()
        if picked:
            self.linkIngestRequested.emit(picked, str(self._language.currentData()))

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
