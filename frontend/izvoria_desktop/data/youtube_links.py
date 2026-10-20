# Path: frontend/izvoria_desktop/data/youtube_links.py

"""
YouTube links for the Video ingest, the same way the web version takes them, with its own code
(web_deploy/common/youtube.py, web_deploy/ingest_worker/fetch_links.py), unchanged: links
are resolved without downloading anything, the picked videos are fetched one at a time as 720p H.264 + AAC MP4
into a folder of their own, and that folder goes through the video pipeline like any other.

The only module in the desktop app that imports `web_deploy.*`. Offered only where yt-dlp is installed
(`available()`): it is in requirements-dev.txt, not requirements.txt, so an appliance never shows it.

The folder is named after the videos picked, so picking the same videos again resumes a stopped batch: files
already there are skipped and yt-dlp continues its partial ones. The files are kept after ingestion, because the
indexed sources point at them and a citation opens them.

Cookies: a Netscape cookies.txt at $YOUTUBE_COOKIES_FILE (default onyx_data/youtube_cookies.txt), when there is one.
yt-dlp saves YouTube's refreshed session back into it, which keeps the sign-in alive.
"""

import hashlib
import importlib.util
import logging
import os
from pathlib import Path
from typing import Callable, List, Optional

from izvoria_desktop.domain.entities import LinkCheck, LinkDownload, ProgressTick, YouTubeVideo
from izvoria_desktop.domain.enums import JobProgressState
from izvoria_desktop.domain.interfaces import IYouTubeLinks
from izvoria_desktop.exceptions.gateway import InvalidRequestError
from izvoria_desktop.utils.i18n import tr

logger = logging.getLogger(__name__)

# The job name its progress ticks carry (utils/progress_stages.py puts it under "Downloading videos")
DOWNLOAD_JOB = "youtube_download"
MAX_VIDEOS = 200
MAX_HEIGHT = 720
DEFAULT_COOKIES = Path("onyx_data") / "youtube_cookies.txt"


def available() -> bool:
    return importlib.util.find_spec("yt_dlp") is not None


def default_root() -> Path:
    """onyx_data/<ENV>/youtube, beside the vault. desktop_app has already moved into the project root."""
    return Path("onyx_data").resolve() / os.environ.get("ENV", "development") / "youtube"


def default_cookies() -> Path:
    return Path(os.environ.get("YOUTUBE_COOKIES_FILE") or DEFAULT_COOKIES).resolve()


class YouTubeLinks(IYouTubeLinks):
    def __init__(self, root: Path, cookies: Optional[Path] = None, extractor=None, downloader=None):
        """`extractor` and `downloader` default to yt-dlp's; tests pass fakes."""
        self._root = Path(root)
        self._cookies = cookies
        self._extractor = extractor
        self._downloader = downloader

    def check(self, links: List[str]) -> LinkCheck:
        from web_deploy.common import youtube

        result = youtube.resolve(links, MAX_VIDEOS, extractor=self._extractor)
        return LinkCheck(
            videos=tuple(YouTubeVideo(v.id, v.title, v.duration) for v in result.videos),
            problems=tuple((p["link"], p["reason"]) for p in result.problems),
            playlist_hints=tuple(result.playlist_hints),
            capped=result.capped,
        )

    def folder_for(self, videos: List[YouTubeVideo]) -> Path:
        digest = hashlib.sha1(",".join(sorted(v.id for v in videos)).encode()).hexdigest()[:12]
        return self._root / digest

    def download(self, videos: List[YouTubeVideo], on_tick: Callable[..., None]) -> LinkDownload:
        from web_deploy.common import youtube
        from web_deploy.ingest_worker import fetch_links

        if not videos:
            raise InvalidRequestError(tr("Pick at least one video to add."))
        folder = self.folder_for(videos)
        batch = [youtube.Video(v.id, v.title, v.duration) for v in videos]
        # A fresh manifest every run, so a video that failed last time (before cookies were added, say) is tried
        # again. What is already downloaded is found by its file, not the manifest.
        youtube.write_manifest(folder, batch)
        download = self._downloader or fetch_links.ytdlp_downloader(MAX_HEIGHT, self._cookies)

        for video in batch:
            if youtube.downloaded(folder, video):
                on_tick(ProgressTick(DOWNLOAD_JOB, JobProgressState.COMPLETED, detail=video.title))
                continue
            on_tick(ProgressTick(DOWNLOAD_JOB, JobProgressState.STARTED, detail=video.title))
            logger.info(f"Downloading {video.url}: {video.title}")
            try:
                download(video, folder, lambda _fraction: None)
                on_tick(ProgressTick(DOWNLOAD_JOB, JobProgressState.COMPLETED, detail=video.title))
            except Exception as exc:
                reason = youtube.reason(exc)
                logger.warning(f"Could not download {video.url}: {reason}")
                youtube.record_errors(folder, {video.file_name: reason})
                on_tick(ProgressTick(DOWNLOAD_JOB, JobProgressState.FAILED, detail=video.title))

        fetch_links._remove_leftovers(folder)
        errors = (youtube.read_manifest(folder) or {}).get("errors", {})
        fetched = sum(1 for video in batch if youtube.downloaded(folder, video))
        if fetched == 0:
            first = next(iter(errors.values()), "")
            logger.error(f"None of the {len(batch)} videos could be downloaded: {first}")
            if youtube.BOT_CHECK in first:
                path = str(self._cookies or DEFAULT_COOKIES)
                if self._cookies and self._cookies.is_file():
                    raise InvalidRequestError(tr(
                        "YouTube refused the downloads: the YouTube cookies at {path} are out of date. Export "
                        "fresh ones and try again.", path=path))
                raise InvalidRequestError(tr(
                    "YouTube refused the downloads until it is signed in. Put a YouTube cookies.txt at {path} "
                    "and try again.", path=path))
            raise InvalidRequestError(tr("None of the videos could be downloaded from YouTube."))
        return LinkDownload(
            folder=str(folder),
            downloaded=fetched,
            errors=tuple(sorted((name, f"could not be downloaded: {reason}") for name, reason in errors.items())),
        )
