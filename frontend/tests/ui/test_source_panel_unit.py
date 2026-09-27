# Canvas: desktop_test_source_panel_v1
# Path: frontend/tests/ui/test_source_panel_unit.py

"""
The source panel replaced an embedded player. Its job is to make a citation checkable: show the
verbatim words, state plainly whether the file is the excerpt or the whole recording, and hand the
file to the desktop.

These tests never actually launch a player -- `QDesktopServices.openUrl` is patched, because a real
call would spawn mpv on the developer's machine.
"""

import uuid
from unittest.mock import patch

import pytest

pytest.importorskip("PySide6")

from onyx_desktop.domain.entities import Evidence  # noqa: E402
from onyx_desktop.domain.enums import MediaKind  # noqa: E402
from onyx_desktop.ui.widgets.source_panel import SourcePanel  # noqa: E402

pytestmark = pytest.mark.gui


def make_evidence(tmp_path, media_kind=MediaKind.CLIP, exists=True, **overrides):
    path = None
    if media_kind is not MediaKind.UNRESOLVED:
        target = tmp_path / "clip.mp4"
        if exists:
            target.write_bytes(b"not really a video")
        path = str(target)

    values = dict(
        doc_number=3,
        chunk_id=uuid.uuid4(),
        source_id=uuid.uuid4(),
        start_sec=2001.0,
        end_sec=2093.0,
        excerpt="SPEAKER_01: there are some very key reference points.",
        speakers=("SPEAKER_01",),
        clip_path=path,
        media_kind=media_kind,
    )
    values.update(overrides)
    return Evidence(**values)


@pytest.fixture
def panel(qt_app):
    widget = SourcePanel()
    # Shown because isVisible() is False for any child of an unshown parent, which would make the
    # visibility assertions below pass vacuously.
    widget.show()
    yield widget
    widget.close()
    widget.deleteLater()


class TestSourcePanel:
    def test_it_starts_empty(self, panel):
        assert panel._card.isVisible() is False

    def test_it_shows_the_verbatim_excerpt_and_timespan(self, panel, tmp_path):
        panel.show_evidence(make_evidence(tmp_path))

        assert panel._chip.text() == "[Doc 3]"
        assert panel._span.text() == "33:21–34:53"
        assert "very key reference points" in panel._excerpt.text()
        assert "SPEAKER_01" in panel._speakers.text()

    def test_a_clip_says_the_file_is_the_excerpt(self, panel, tmp_path):
        panel.show_evidence(make_evidence(tmp_path, media_kind=MediaKind.CLIP))

        assert panel._open.isEnabled() is True
        assert panel._open.text() == "Open clip"
        assert "is the cited excerpt" in panel._kind_note.text()

    def test_a_full_source_tells_the_user_where_to_skip_to(self, panel, tmp_path):
        """
        A system player cannot be told to start at a timestamp, so the panel has to say the number
        out loud instead of quietly opening the file at 0:00.
        """
        panel.show_evidence(make_evidence(tmp_path, media_kind=MediaKind.FULL_SOURCE))

        assert panel._open.text() == "Open full recording"
        assert "whole recording" in panel._kind_note.text()
        assert "33:21–34:53" in panel._kind_note.text()

    def test_an_unresolved_citation_disables_opening_but_keeps_the_excerpt(self, panel, tmp_path):
        panel.show_evidence(make_evidence(tmp_path, media_kind=MediaKind.UNRESOLVED))

        assert panel._open.isEnabled() is False
        assert panel._open.toolTip()          # the reason is stated, not left blank
        # The words are the evidence; losing the file must not hide them.
        assert "very key reference points" in panel._excerpt.text()

    def test_a_chunk_with_no_stored_text_says_so(self, panel, tmp_path):
        panel.show_evidence(make_evidence(tmp_path, excerpt=""))

        assert "no transcript text" in panel._excerpt.text()

    def test_opening_hands_the_file_to_the_desktop(self, panel, tmp_path):
        evidence = make_evidence(tmp_path)
        panel.show_evidence(evidence)

        with patch("onyx_desktop.ui.system_open.QDesktopServices.openUrl",
                   return_value=True) as opened:
            panel._on_open()

        assert opened.call_count == 1
        assert opened.call_args.args[0].toLocalFile() == evidence.clip_path
        assert panel._problem.isVisible() is False

    def test_a_missing_file_is_reported_rather_than_silently_opened(self, panel, tmp_path):
        panel.show_evidence(make_evidence(tmp_path, exists=False))

        failures = []
        panel.openFailed.connect(failures.append)
        with patch("onyx_desktop.ui.system_open.QDesktopServices.openUrl") as opened:
            panel._on_open()

        opened.assert_not_called()
        assert panel._problem.isVisible() is True
        assert "no longer in the vault" in failures[0]

    def test_a_desktop_with_no_handler_is_reported(self, panel, tmp_path):
        panel.show_evidence(make_evidence(tmp_path))

        failures = []
        panel.openFailed.connect(failures.append)
        with patch("onyx_desktop.ui.system_open.QDesktopServices.openUrl",
                   return_value=False):
            panel._on_open()

        assert panel._problem.isVisible() is True
        assert "no default application" in failures[0]

    def test_copy_puts_the_timespan_on_the_clipboard(self, panel, tmp_path):
        from PySide6.QtGui import QGuiApplication

        panel.show_evidence(make_evidence(tmp_path))
        panel._on_copy()

        assert QGuiApplication.clipboard().text() == "33:21–34:53"

    def test_clear_hides_the_card_again(self, panel, tmp_path):
        panel.show_evidence(make_evidence(tmp_path))
        panel.clear()

        assert panel._card.isVisible() is False


class TestImageSource:
    def test_an_image_shows_its_preview_and_warns_the_excerpt_is_a_description(self, panel, tmp_path):
        from PySide6.QtGui import QImage
        path = tmp_path / "scan.png"
        QImage(200, 400, QImage.Format.Format_RGB32).save(str(path))

        panel.show_evidence(make_evidence(tmp_path, media_kind=MediaKind.IMAGE, clip_path=str(path),
                                          location="image"))

        assert panel._open.text() == "Open image" and panel._open.isEnabled()
        assert "machine description" in panel._kind_note.text()
        assert panel._thumbnail.isVisible() and panel._thumbnail.pixmap().height() == 320

    def test_the_preview_is_cleared_for_the_next_non_image(self, panel, tmp_path):
        from PySide6.QtGui import QImage
        path = tmp_path / "scan.png"
        QImage(20, 20, QImage.Format.Format_RGB32).save(str(path))
        panel.show_evidence(make_evidence(tmp_path, media_kind=MediaKind.IMAGE, clip_path=str(path)))

        panel.show_evidence(make_evidence(tmp_path, media_kind=MediaKind.CLIP))

        assert panel._thumbnail.isVisible() is False
