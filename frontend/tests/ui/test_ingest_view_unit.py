# Path: frontend/tests/ui/test_ingest_view_unit.py

import uuid

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402

from izvoria_desktop.domain.entities import (  # noqa: E402
    IngestionSummary, LanguageOption, LinkCheck, ProgressTick, YouTubeVideo,
)
from izvoria_desktop.domain.enums import IngestMode, JobProgressState  # noqa: E402
from izvoria_desktop.ui.widgets.stage_list import StageState  # noqa: E402
from izvoria_desktop.ui.views.ingest_view import IngestView  # noqa: E402

pytestmark = pytest.mark.gui

LANGUAGES = [LanguageOption("en", "English"), LanguageOption("sr", "Srpski")]


@pytest.fixture
def view(qt_app):
    instance = IngestView()
    instance.set_languages(LANGUAGES, "en")
    yield instance
    instance.deleteLater()


class TestFileLanguage:
    def _request(self, view):
        requests = []
        view.ingestRequested.connect(lambda *args: requests.append(args))
        view._path.setText("/media/batch")
        view._on_start()
        return requests

    def test_the_request_carries_the_language_as_a_plain_code(self, view):
        view._language.setCurrentIndex(view._language.findData("sr"))

        [(path, mode, language)] = self._request(view)

        assert (path, mode, language) == ("/media/batch", IngestMode.VIDEO, "sr")
        # Rebuilt at the boundary: a QVariant must never reach the gateway (the QComboBox gotcha)
        assert type(language) is str and type(mode) is IngestMode

    def test_defaults_to_the_active_language_and_follows_a_switch(self, view):
        assert self._request(view)[0][2] == "en"

        view.select_language("sr")

        assert self._request(view)[0][2] == "sr"

    def test_languages_are_listed_by_their_own_names(self, view):
        assert [view._language.itemText(i) for i in range(view._language.count())] == [
            "English", "Srpski"]

    def test_the_picker_is_locked_while_busy(self, view):
        view.set_busy(True)
        assert not view._language.isEnabled()
        view.set_busy(False)
        assert view._language.isEnabled()


class TestSteps:
    """Progress is a few general steps; the engine's own jobs and log never reach the screen."""

    SOURCE_A = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    SOURCE_B = uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")

    def _tick(self, phase, source=None, state=JobProgressState.COMPLETED):
        return ProgressTick(phase=phase, state=state, source_id=source)

    def _start(self, view, mode=IngestMode.VIDEO):
        view._mode.setCurrentIndex(view._mode.findData(mode))
        view.start_run()

    def test_a_run_lists_the_general_steps_for_its_kind(self, view):
        self._start(view, IngestMode.DOCUMENT)
        assert view._stages.keys == ["preparing", "documents", "context", "indexing"]
        assert view._stages.state_of("preparing") is StageState.ACTIVE
        assert view._stages.state_of("indexing") is StageState.WAITING

    def test_a_job_moves_its_step_and_finishes_the_ones_before(self, view):
        self._start(view)
        view.apply_tick(self._tick("file_ingest", self.SOURCE_A))
        view.apply_tick(self._tick("file_ingest", self.SOURCE_B))

        view.apply_tick(self._tick("transcribe", self.SOURCE_A))

        assert view._stages.state_of("preparing") is StageState.DONE
        assert view._stages.state_of("audio") is StageState.ACTIVE
        # 1 of 2 files x 7 audio jobs
        assert 0 < view._stages.value_of("audio") < 200
        assert view._phase_label.text() == "Step 2 of 5 · 2 files"

    def test_a_skipped_step_is_finished_when_a_later_one_starts(self, view):
        self._start(view)
        view.apply_tick(self._tick("contextual_retrieval", self.SOURCE_A))
        view.apply_tick(self._tick("rag_indexing", self.SOURCE_A))  # no visual cues in this batch

        assert view._stages.state_of("visuals") is StageState.DONE
        assert view._stages.state_of("indexing") is StageState.ACTIVE

    def test_a_job_outside_every_step_is_ignored(self, view):
        self._start(view)
        view.apply_tick(self._tick("rag_retrieval"))  # a live search job, not part of a batch
        assert view._stages.state_of("preparing") is StageState.ACTIVE
        assert view._phase_label.text() == "Step 1 of 5"

    def test_finishing_completes_every_step(self, view):
        self._start(view)
        view.show_summary(IngestionSummary(target_path="/media", scanned=2, succeeded=2))
        assert all(view._stages.state_of(key) is StageState.DONE for key in view._stages.keys)

    def test_a_failure_marks_where_it_stopped(self, view):
        self._start(view)
        view.apply_tick(self._tick("transcribe", self.SOURCE_A))
        view.show_failure("The appliance could not complete that request.")
        assert view._stages.state_of("audio") is StageState.FAILED
        assert view._stages.state_of("indexing") is StageState.WAITING

    def test_file_problems_are_reported_in_plain_words(self, view):
        self._start(view)
        view.show_summary(IngestionSummary(
            target_path="/media", scanned=1, failed=1,
            errors=(("/media/batch/talk.mp4",
                     "Whisper heard 'de' (probability 0.97) in source 1234, not English, so it was not transcribed"),),
        ))
        text = view._report._errors.text()
        assert text == "• talk.mp4: Its spoken language does not match the one selected."

    def test_a_run_reassures_that_a_still_bar_is_normal_and_the_notice_goes_when_done(self, view):
        assert not view._patience.isVisibleTo(view)
        self._start(view)
        assert view._patience.isVisibleTo(view)
        assert "no need to restart" in view._patience.text()

        view.show_summary(IngestionSummary(target_path="/media", scanned=1, succeeded=1))
        assert not view._patience.isVisibleTo(view)

    def test_there_is_no_engine_log_on_screen(self, view):
        assert not hasattr(view, "_log")
        assert not hasattr(view, "append_log")


class TestYouTubeLinks:
    VIDEOS = (YouTubeVideo("aaaaaaaaaaa", "Lecture 1", 2950.0), YouTubeVideo("bbbbbbbbbbb", "Lecture 2", None))

    def test_offered_for_video_batches_only_and_only_where_available(self, view):
        assert view._links.isHidden()
        view.set_links_available(True)
        assert not view._links.isHidden()
        view._mode.setCurrentIndex(view._mode.findData(IngestMode.AUDIO))
        assert view._links.isHidden()

    def test_check_sends_one_link_per_line(self, view):
        requests = []
        view.linksCheckRequested.connect(requests.append)
        view._link_text.setPlainText("  https://youtu.be/aaaaaaaaaaa \n\nhttps://www.youtube.com/playlist?list=PLx\n")

        view._on_check_links()

        assert requests == [["https://youtu.be/aaaaaaaaaaa", "https://www.youtube.com/playlist?list=PLx"]]
        assert not view._check_links.isEnabled()   # until the answer comes back

    def test_only_the_ticked_videos_are_added_in_the_picked_language(self, view):
        requests = []
        view.linkIngestRequested.connect(lambda *args: requests.append(args))
        view.show_link_check(LinkCheck(videos=self.VIDEOS))
        view._video_list.item(0).setCheckState(Qt.CheckState.Unchecked)
        view._language.setCurrentIndex(view._language.findData("sr"))

        view._on_add_videos()

        assert requests == [([self.VIDEOS[1]], "sr")]

    def test_nothing_ticked_cannot_be_added(self, view):
        view.show_link_check(LinkCheck(videos=self.VIDEOS[:1]))
        assert view._add_videos.isEnabled()
        view._video_list.item(0).setCheckState(Qt.CheckState.Unchecked)
        assert not view._add_videos.isEnabled()

    def test_links_that_added_nothing_are_listed_with_youtubes_reason(self, view):
        view.show_link_check(LinkCheck(problems=(("https://youtu.be/x", "Private video"),)))
        assert "https://youtu.be/x: Private video" in view._links_problems.text()
        assert view._video_list.isHidden()

    def test_a_batch_from_links_downloads_first_and_its_bar_counts_videos(self, view):
        view.start_run(videos=2)
        assert view._stages.keys[0] == "downloading"

        view.apply_tick(ProgressTick("youtube_download", JobProgressState.COMPLETED))

        assert view._fraction("downloading") == 0.5
        assert not view._files   # a download is not a file of the batch yet
