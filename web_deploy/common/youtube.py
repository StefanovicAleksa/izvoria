"""
YouTube links for the video pipeline: which links are accepted, what videos they name, and the batch's manifest.

Shared by the API (which checks links and queues the batch) and the ingestion worker (which downloads it before the
GPU is taken). Web deployment only: the on-premise appliance never reaches the internet.

A link is resolved with yt-dlp's flat extraction, which reads a playlist's entries without downloading anything, so
the user sees exactly which videos a batch holds before queueing it. The batch folder then carries a `.youtube.json`
manifest (a dotfile, so the upload filters never count it as media) listing the videos to fetch, and later which ones
failed and why (keyed by file name, as the ingestion report keys its failures).
"""

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

MANIFEST = ".youtube.json"

_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
_LIST_ID = re.compile(r"^[A-Za-z0-9_-]{10,64}$")
_HOSTS = {"youtube.com", "m.youtube.com", "music.youtube.com", "youtube-nocookie.com"}
_SHORT_HOST = "youtu.be"
# /shorts/<id>, /embed/<id>, /live/<id>, /v/<id>
_PATH_ID = re.compile(r"^/(?:shorts|embed|live|v)/([A-Za-z0-9_-]{11})(?:[/?#]|$)")
# Placeholders yt-dlp's flat playlist extraction returns for entries nobody can watch
_UNAVAILABLE_TITLES = {"[private video]", "[deleted video]", "[unavailable video]"}
# Everything a file name can't hold on the box (path separators, Windows' reserved characters, control characters)
_UNSAFE_IN_NAME = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
MAX_TITLE_IN_NAME = 150


class LinkError(ValueError):
    """A link that is not a YouTube video or playlist; the message is shown to the user as it is."""


@dataclass(frozen=True)
class Link:
    kind: str  # "video" or "playlist"
    id: str
    # This video sits in a playlist (watch?v=…&list=…): only the video is added, and the user is told so
    in_playlist: bool = False

    @property
    def url(self) -> str:
        if self.kind == "playlist":
            return f"https://www.youtube.com/playlist?list={self.id}"
        return f"https://www.youtube.com/watch?v={self.id}"


@dataclass(frozen=True)
class Video:
    id: str
    title: str
    duration: Optional[float] = None

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.id}"

    @property
    def file_name(self) -> str:
        """`<title> [<id>].mp4`: the title is the source's name in citations, the id finds the file again."""
        title = " ".join(_UNSAFE_IN_NAME.sub(" ", self.title).split()).strip(". ") or "YouTube video"
        if len(title) > MAX_TITLE_IN_NAME:
            title = title[:MAX_TITLE_IN_NAME].rsplit(" ", 1)[0].rstrip(" .,-–") or title[:MAX_TITLE_IN_NAME]
        return f"{title} [{self.id}].mp4"


@dataclass
class Resolution:
    videos: List[Video] = field(default_factory=list)
    # One per link that added nothing: {"link": what was typed, "reason": why}
    problems: List[Dict[str, str]] = field(default_factory=list)
    # Links that named a video inside a playlist: only the video was added
    playlist_hints: List[str] = field(default_factory=list)
    # True when the batch was cut at the cap
    capped: bool = False

    def public(self) -> dict:
        return {"videos": [asdict(v) for v in self.videos], "problems": self.problems,
                "playlist_hints": self.playlist_hints, "capped": self.capped}


def parse_link(raw: str) -> Link:
    text = (raw or "").strip()
    if not text:
        raise LinkError("The link is empty.")
    if "://" not in text:
        text = "https://" + text
    parsed = urlparse(text)
    host = (parsed.hostname or "").lower().removeprefix("www.")
    query = parse_qs(parsed.query)

    if host == _SHORT_HOST:
        candidate = parsed.path.strip("/").split("/")[0]
        if _VIDEO_ID.match(candidate):
            return Link("video", candidate, in_playlist="list" in query)
        raise LinkError("This youtu.be link has no video in it.")

    if host not in _HOSTS:
        raise LinkError("Only YouTube links can be added.")

    if parsed.path.rstrip("/") == "/playlist":
        list_id = (query.get("list") or [""])[0]
        if _LIST_ID.match(list_id):
            return Link("playlist", list_id)
        raise LinkError("This playlist link has no playlist id.")

    if parsed.path.rstrip("/") == "/watch":
        video_id = (query.get("v") or [""])[0]
        if _VIDEO_ID.match(video_id):
            return Link("video", video_id, in_playlist="list" in query)
        raise LinkError("This link has no video id.")

    match = _PATH_ID.match(parsed.path)
    if match:
        return Link("video", match.group(1))

    raise LinkError("This YouTube link is not a video or a playlist (channels and searches can't be added).")


Extractor = Callable[[str], Dict[str, Any]]


def resolve(raw_links: List[str], max_videos: int, extractor: Optional[Extractor] = None) -> Resolution:
    """
    The videos the links name, in order, each once, at most `max_videos`. Nothing is downloaded. A link that can't
    be read, and entries that are private, deleted, live or not yet aired, are reported rather than queued.
    """
    extract = extractor or ytdlp_extractor()
    result = Resolution()
    seen = set()

    for raw in raw_links:
        raw = (raw or "").strip()
        if not raw:
            continue
        try:
            link = parse_link(raw)
        except LinkError as exc:
            result.problems.append({"link": raw, "reason": str(exc)})
            continue
        if link.in_playlist:
            result.playlist_hints.append(raw)
        try:
            info = extract(link.url)
        except Exception as exc:
            result.problems.append({"link": raw, "reason": reason(exc)})
            continue

        entries = info.get("entries") if link.kind == "playlist" else [info]
        added = 0
        for entry in entries or []:
            if not entry:
                continue
            video_id = str(entry.get("id") or "")
            title = str(entry.get("title") or "").strip()
            if not _VIDEO_ID.match(video_id) or title.lower() in _UNAVAILABLE_TITLES:
                continue
            if entry.get("live_status") in ("is_live", "is_upcoming", "post_live"):
                continue
            if video_id in seen:
                continue
            if len(result.videos) >= max_videos:
                result.capped = True
                break
            seen.add(video_id)
            duration = entry.get("duration")
            result.videos.append(Video(video_id, title or video_id, float(duration) if duration else None))
            added += 1
        if added == 0 and not result.capped:
            result.problems.append({"link": raw, "reason": _empty_reason(link, info)})

    return result


def ytdlp_extractor() -> Extractor:
    import yt_dlp

    options = {
        "quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True,
        # A playlist's entries are listed without opening each video; a single video is read whole (for live_status)
        "extract_flat": "in_playlist", "socket_timeout": 30,
    }

    def extract(url: str) -> Dict[str, Any]:
        with yt_dlp.YoutubeDL(options) as client:
            return client.extract_info(url, download=False) or {}

    return extract


def reason(exc: Exception) -> str:
    """yt-dlp's message, for the user without its "ERROR: [youtube] <id>:" prefix: "Private video", "Video unavailable"."""
    message = re.sub(r"^ERROR:\s*(\[[^\]]+\]\s*)?([A-Za-z0-9_-]+:\s*)?", "", str(exc).strip())
    return message.split("\n")[0][:200] or "YouTube could not be reached."


def _empty_reason(link: Link, info: Dict[str, Any]) -> str:
    if link.kind == "playlist":
        return "The playlist has no videos that can be downloaded."
    status = info.get("live_status")
    if status in ("is_live", "is_upcoming", "post_live"):
        return "Live streams can't be added until they have ended and been processed."
    return "The video can't be downloaded."


# ---------- the batch manifest ----------

def new_manifest(videos: List[Video]) -> Dict[str, Any]:
    return {"videos": [asdict(v) for v in videos], "errors": {}}


def write_manifest(folder: Path, videos: List[Video]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    _write(folder, new_manifest(videos))


def place_manifest(folder: Path, manifest: Dict[str, Any]) -> None:
    """
    Puts the job row's manifest into the batch folder the download works in, unless one is already there: a download
    a stop interrupted has recorded its failures in it, and those must survive.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if read_manifest(folder) is None:
        _write(folder, manifest)


def read_manifest(folder: Path) -> Optional[Dict[str, Any]]:
    path = Path(folder) / MANIFEST
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_videos(manifest: Dict[str, Any]) -> List[Video]:
    return [Video(v["id"], v["title"], v.get("duration")) for v in manifest.get("videos", [])]


def record_errors(folder: Path, errors: Dict[str, str]) -> None:
    """Download failures, keyed by the video's file name (as the ingestion report keys its own)."""
    manifest = read_manifest(folder) or {"videos": [], "errors": {}}
    manifest["errors"] = {**manifest.get("errors", {}), **errors}
    _write(Path(folder), manifest)


def downloaded(folder: Path, video: Video) -> bool:
    return (Path(folder) / video.file_name).is_file()


def pending(folder: Path) -> List[Video]:
    """Videos of the batch neither downloaded nor given up on."""
    manifest = read_manifest(folder)
    if manifest is None:
        return []
    failed = set(manifest.get("errors", {}))
    return [v for v in manifest_videos(manifest) if not downloaded(folder, v) and v.file_name not in failed]


def _write(folder: Path, manifest: Dict[str, Any]) -> None:
    path = folder / MANIFEST
    partial = folder / f"{MANIFEST}.part"
    partial.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    partial.replace(path)
