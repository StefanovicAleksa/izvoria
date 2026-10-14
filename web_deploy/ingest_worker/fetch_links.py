"""
Downloads a YouTube batch's videos before it is ingested.

The supervisor runs this as a child process, before it asks for the GPU, so videos download while chat still
answers (on dev hardware the live models only step aside for the ingestion itself). A child, like the ingestion: the
supervisor can stop it at any moment, and yt-dlp resumes its partial files when the job runs again.

One video at a time, up to LINK_MAX_HEIGHT (720p by default), as H.264 + AAC MP4: what plays in every browser when a
citation is opened, and sharp enough for slides and charts. A video that fails is recorded in the batch's manifest
and the rest go on; the ingestion report lists it with YouTube's reason. A batch where nothing could be downloaded
fails here, before the GPU is ever taken.

Run: python -m web_deploy.ingest_worker.fetch_links <job id>
"""

import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Optional

from web_deploy.common import youtube
from web_deploy.common.youtube import Video
from web_deploy.ingest_worker.queue import IngestQueue

logger = logging.getLogger(__name__)

PROGRESS_INTERVAL_SECONDS = 2.0

# (video, folder, on_fraction) -> None; raises with a reason the user can read
Downloader = Callable[[Video, Path, Callable[[float], None]], None]


COOKIES_HINT = " The owner can add YouTube cookies in the console (GPU server → YouTube downloads)."


def fetch(job_id: str, queue: IngestQueue, downloader: Downloader, clock=time.monotonic, wall=time.time,
          cookies: Optional[Path] = None) -> int:
    """Downloads every pending video of the job's batch. Returns 0 when at least one video is there, else 1."""
    job = queue.get(job_id)
    if job is None:
        logger.error(f"Job {job_id} does not exist.")
        return 2
    folder = Path(job.batch_dir)
    manifest = youtube.read_manifest(folder)
    videos = youtube.manifest_videos(manifest) if manifest else []
    total = len(videos)
    last = [float("-inf")]

    def report(done: float, title: Optional[str], force: bool = False) -> None:
        now = clock()
        if not force and now - last[0] < PROGRESS_INTERVAL_SECONDS:
            return
        last[0] = now
        queue.record_progress(job_id, {"phase": "downloading", "state": "started", "detail": title,
                                       "done": round(done, 3), "total": total, "updated_at": wall()})

    for video in youtube.pending(folder):
        finished = sum(1 for v in videos if youtube.downloaded(folder, v))
        report(finished, video.title, force=True)
        logger.info(f"Downloading {video.url}: {video.title}")
        try:
            downloader(video, folder, lambda fraction, base=finished, title=video.title: report(base + fraction, title))
        except Exception as exc:
            reason = youtube.reason(exc)
            logger.warning(f"Could not download {video.url}: {reason}")
            youtube.record_errors(folder, {video.file_name: reason})

    _remove_leftovers(folder)
    fetched = [v for v in videos if youtube.downloaded(folder, v)]
    report(len(fetched), None, force=True)
    if not fetched:
        errors = (youtube.read_manifest(folder) or {}).get("errors", {})
        first = next(iter(errors.values()), "no reason was given")
        hint = COOKIES_HINT if youtube.BOT_CHECK in first and (cookies is None or not cookies.is_file()) else ""
        queue.fail(job_id, f"None of the {total} videos could be downloaded: {first}{hint}")
        return 1
    return 0


# What a stopped or failed yt-dlp download leaves beside the finished MP4s: the separate video and audio streams
# ("Title [id].f137.mp4"), partial files and its bookkeeping. The video pipeline would take a stream for a video.
_LEFTOVER = re.compile(r"\.f\d+(?:-\d+)?\.\w+$|\.(?:part|ytdl|temp)(?:-Frag\d+)?$|\.temp\.\w+$")


def _remove_leftovers(folder: Path) -> None:
    for path in folder.iterdir():
        if path.is_file() and _LEFTOVER.search(path.name):
            path.unlink(missing_ok=True)


def ytdlp_downloader(max_height: int, cookies: Optional[Path] = None) -> Downloader:
    """
    yt-dlp, one video at a time, as MP4 named after the video (`Video.file_name`). With `cookies` (a Netscape
    cookies.txt, when it exists), as that YouTube account: the rented box's data-centre IP needs one.
    """
    import yt_dlp

    # H.264 + AAC first (plays everywhere); otherwise the best at or under the height, remuxed into MP4
    selector = (f"bv*[height<={max_height}][vcodec^=avc1]+ba[ext=m4a]/b[height<={max_height}][ext=mp4]"
                f"/bv*[height<={max_height}]+ba/b[height<={max_height}]/b")

    def download(video: Video, folder: Path, on_fraction: Callable[[float], None]) -> None:
        stem = Path(video.file_name).stem.replace("%", "%%")

        def hook(status: Dict) -> None:
            total = status.get("total_bytes") or status.get("total_bytes_estimate")
            if status.get("status") == "downloading" and total:
                on_fraction(min(0.99, status.get("downloaded_bytes", 0) / total))

        options = {
            "format": selector,
            "merge_output_format": "mp4",
            "postprocessors": [{"key": "FFmpegVideoRemuxer", "preferedformat": "mp4"}],
            "outtmpl": str(folder / f"{stem}.%(ext)s"),
            "noplaylist": True, "quiet": True, "no_warnings": True, "noprogress": True,
            "retries": 5, "fragment_retries": 5, "socket_timeout": 30, "continuedl": True,
            "progress_hooks": [hook],
            "js_runtimes": _js_runtimes(),
        }
        if cookies is not None and cookies.is_file():
            options["cookiefile"] = str(cookies)
        with yt_dlp.YoutubeDL(options) as client:
            client.download([video.url])
        if not youtube.downloaded(folder, video):
            raise RuntimeError("The download finished without an MP4 file.")

    return download


def _js_runtimes() -> Dict[str, Dict]:
    """
    YouTube's player challenge needs a JavaScript runtime, and yt-dlp enables only deno unless told otherwise. deno
    comes from PyPI (web_deploy/api/requirements.txt), installed beside this Python, which a worker started by
    supervisord may not have on its PATH; then whatever is on the PATH.
    """
    beside = Path(sys.executable).with_name("deno")
    if beside.is_file():
        return {"deno": {"path": str(beside)}}
    found = {name: {} for name in ("deno", "node", "bun") if shutil.which(name)}
    return found or {"deno": {}}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from web_deploy.common.box_credentials import install_from_env
    install_from_env()  # the hosted box: the database password comes from the gateway, never from the environment
    from web_deploy.api.config import ApiSettings
    settings = ApiSettings.from_env()
    queue = IngestQueue.connect(settings.control_database_url)
    cookies = Path(os.environ.get("YOUTUBE_COOKIES_FILE", "/workspace/web/youtube_cookies.txt"))
    sys.exit(fetch(sys.argv[1], queue, ytdlp_downloader(settings.link_max_height, cookies), cookies=cookies))


if __name__ == "__main__":
    main()
