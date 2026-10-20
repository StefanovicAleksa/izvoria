# Path: frontend/tests/data/test_youtube_links_unit.py

import pytest

from izvoria_desktop.data.youtube_links import DOWNLOAD_JOB, YouTubeLinks
from izvoria_desktop.domain.entities import YouTubeVideo
from izvoria_desktop.domain.enums import JobProgressState
from izvoria_desktop.exceptions.gateway import InvalidRequestError
from web_deploy.common import youtube

VIDEOS = [YouTubeVideo("aaaaaaaaaaa", "Lecture 1: Probability", 2950.0),
          YouTubeVideo("bbbbbbbbbbb", "Lecture 2: Story Proofs", 3000.0)]
PLAYLIST = "https://www.youtube.com/playlist?list=PL2SOU6wwxB0uwwH80KTQ6ht66KWxbzTIo"


def as_web(video):
    return youtube.Video(video.id, video.title, video.duration)


class FakeDownloader:
    """Writes the MP4 yt-dlp would, or raises for the ids in `fail` with YouTube's message."""

    def __init__(self, fail=None):
        self.fail = fail or {}
        self.calls = []

    def __call__(self, video, folder, on_fraction):
        self.calls.append(video.id)
        if video.id in self.fail:
            raise RuntimeError(f"ERROR: [youtube] {video.id}: {self.fail[video.id]}")
        (folder / video.file_name).write_bytes(b"mp4")


def links(tmp_path, downloader, cookies=None, extractor=None):
    return YouTubeLinks(tmp_path / "youtube", cookies=cookies, extractor=extractor, downloader=downloader)


class TestCheck:
    def test_a_playlist_lists_its_videos_and_unreadable_links_are_reported(self, tmp_path):
        def extractor(url):
            assert url == PLAYLIST
            return {"entries": [{"id": "aaaaaaaaaaa", "title": "Lecture 1: Probability", "duration": 2950},
                                {"id": "ccccccccccc", "title": "[Private video]"}]}

        check = links(tmp_path, FakeDownloader(), extractor=extractor).check([PLAYLIST, "https://vimeo.com/1"])

        assert check.videos == (YouTubeVideo("aaaaaaaaaaa", "Lecture 1: Probability", 2950.0),)
        assert check.problems == (("https://vimeo.com/1", "Only YouTube links can be added."),)
        assert check.capped is False


class TestDownload:
    def test_every_video_is_downloaded_into_one_folder_with_a_tick_each(self, tmp_path):
        downloader, ticks = FakeDownloader(), []

        result = links(tmp_path, downloader).download(VIDEOS, ticks.append)

        assert result.downloaded == 2 and result.errors == ()
        assert sorted(p.name for p in (tmp_path / "youtube").iterdir()) == [result.folder.rsplit("/", 1)[-1]]
        assert {(t.phase, t.state) for t in ticks} == {(DOWNLOAD_JOB, JobProgressState.STARTED),
                                                       (DOWNLOAD_JOB, JobProgressState.COMPLETED)}
        assert all(t.source_id is None for t in ticks)

    def test_the_same_videos_get_the_same_folder_and_resume_there(self, tmp_path):
        first = links(tmp_path, FakeDownloader(fail={"bbbbbbbbbbb": "Video unavailable"}))
        folder = first.download(VIDEOS, lambda tick: None).folder
        again = FakeDownloader()

        result = links(tmp_path, again).download(list(reversed(VIDEOS)), lambda tick: None)

        assert result.folder == folder
        # The one already downloaded is skipped; the one that failed is tried again
        assert again.calls == ["bbbbbbbbbbb"]
        assert result.downloaded == 2 and result.errors == ()

    def test_a_video_that_fails_is_listed_and_the_rest_go_on(self, tmp_path):
        result = links(tmp_path, FakeDownloader(fail={"aaaaaaaaaaa": "Private video"})).download(
            VIDEOS, lambda tick: None)

        assert result.downloaded == 1
        assert result.errors == ((as_web(VIDEOS[0]).file_name, "could not be downloaded: Private video"),)

    def test_leftover_streams_are_removed_so_the_pipeline_never_takes_one_for_a_video(self, tmp_path):
        service = links(tmp_path, FakeDownloader())
        folder = service.folder_for(VIDEOS)
        folder.mkdir(parents=True)
        (folder / "Lecture 1 [aaaaaaaaaaa].f137.mp4").write_bytes(b"stream")

        service.download(VIDEOS, lambda tick: None)

        assert not (folder / "Lecture 1 [aaaaaaaaaaa].f137.mp4").exists()

    def test_nothing_downloaded_is_a_failure(self, tmp_path):
        downloader = FakeDownloader(fail={v.id: "Video unavailable" for v in VIDEOS})

        with pytest.raises(InvalidRequestError, match="None of the videos could be downloaded"):
            links(tmp_path, downloader).download(VIDEOS, lambda tick: None)

    @pytest.mark.parametrize("cookies_exist, expected", [
        (False, "Put a YouTube cookies.txt at"),
        (True, "are out of date"),
    ])
    def test_a_bot_check_says_what_to_do_about_the_cookies(self, tmp_path, cookies_exist, expected):
        cookies = tmp_path / "youtube_cookies.txt"
        if cookies_exist:
            cookies.write_text("# Netscape HTTP Cookie File\n")
        downloader = FakeDownloader(fail={v.id: "Sign in to confirm you're not a bot" for v in VIDEOS})

        with pytest.raises(InvalidRequestError, match=expected) as raised:
            links(tmp_path, downloader, cookies=cookies).download(VIDEOS, lambda tick: None)
        assert str(cookies) in raised.value.user_message

    def test_an_empty_pick_is_refused(self, tmp_path):
        with pytest.raises(InvalidRequestError):
            links(tmp_path, FakeDownloader()).download([], lambda tick: None)
