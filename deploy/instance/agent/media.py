"""
Turning an offered source into something a browser can play or show.

A clip is cut with `-c copy`, so it keeps whatever codec the recording had; an audio source may be WMA or AMR; an
image may be TIFF. When the browser can't take the original, ffmpeg makes a copy it can, once, into the cache.
Cutting a cited passage out of a whole recording works the same way. All of it runs one job at a time at low
priority, so it never fights a running ingestion for the CPU.
"""

import asyncio
import codecs
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, List, Optional, Sequence, Tuple

from charset_normalizer import from_bytes

from deploy.instance.agent.sources import Offer, SourceKind, Variant

logger = logging.getLogger(__name__)

FFMPEG_TIMEOUT_SECONDS = 3600.0
PROBE_TIMEOUT_SECONDS = 30.0
# The cache lives on the instance disk, which is billed while the box is stopped too. Past this, the files least
# recently made go first; anything evicted is simply made again on the next click.
CACHE_MAX_BYTES = 5 * 1024 ** 3

# What every current browser plays, per container. Anything else is transcoded to H.264/AAC MP4 or AAC M4A.
_PLAYABLE_VIDEO = {
    ".mp4": ({"h264", "av1", "vp9"}, {None, "aac", "mp3", "opus"}),
    ".webm": ({"vp8", "vp9", "av1"}, {None, "opus", "vorbis"}),
}
_PLAYABLE_AUDIO = {
    ".mp3": {"mp3"},
    ".wav": {"pcm_s16le", "pcm_s24le", "pcm_u8", "pcm_f32le"},
    ".flac": {"flac"},
    ".m4a": {"aac"},
    ".aac": {"aac"},
    ".ogg": {"vorbis", "opus"},
    ".opus": {"opus"},
}
_PLAYABLE_IMAGES = {".jpg", ".jpeg", ".png", ".webp"}

_H264 = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p"]
_AAC = ["-c:a", "aac", "-b:a", "128k"]

Runner = Callable[[Sequence[str], float], Awaitable[Tuple[int, bytes]]]


class MediaError(RuntimeError):
    pass


@dataclass(frozen=True)
class Probe:
    video_codec: Optional[str]
    audio_codec: Optional[str]


@dataclass(frozen=True)
class Prepared:
    path: Path
    mime: str


def parse_probe(raw: bytes) -> Probe:
    try:
        streams = json.loads(raw or b"{}").get("streams", [])
    except ValueError as exc:
        raise MediaError("ffprobe returned something that isn't JSON.") from exc

    def first(codec_type: str) -> Optional[str]:
        # Cover art in an MP3 or M4A is a "video" stream; it doesn't make the file a video.
        for stream in streams:
            if stream.get("codec_type") == codec_type and not (stream.get("disposition") or {}).get("attached_pic"):
                return stream.get("codec_name")
        return None

    return Probe(video_codec=first("video"), audio_codec=first("audio"))


def browser_playable(kind: SourceKind, suffix: str, probe: Probe) -> bool:
    suffix = suffix.lower()
    if kind is SourceKind.VIDEO:
        if suffix not in _PLAYABLE_VIDEO or probe.video_codec is None:
            return False
        video, audio = _PLAYABLE_VIDEO[suffix]
        return probe.video_codec in video and probe.audio_codec in audio
    if kind is SourceKind.AUDIO:
        return probe.audio_codec in _PLAYABLE_AUDIO.get(suffix, set())
    return False


def decode_text(data: bytes) -> str:
    """
    The appliance's own decoding (document_conversion's CharsetTextDecoder) and line breaks as it counts them, so
    line N here is line N of the citation.
    """
    if data.startswith(codecs.BOM_UTF8):
        text = data[len(codecs.BOM_UTF8):].decode("utf-8")
    else:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            best = from_bytes(data).best()
            text = str(best) if best is not None else data.decode("utf-8", errors="replace")
    return text.replace("\r\n", "\n").replace("\r", "\n")


async def run_process(command: Sequence[str], timeout: float) -> Tuple[int, bytes]:
    process = await asyncio.create_subprocess_exec(
        *command, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        raise MediaError(f"{command[0]} took longer than {timeout:.0f} s.")
    if process.returncode != 0:
        logger.warning(f"{command[0]} failed ({process.returncode}): {stderr.decode(errors='replace')[-2000:]}")
    return process.returncode, stdout


class MediaPreparer:
    def __init__(self, cache_root: Path, runner: Runner = run_process, max_bytes: int = CACHE_MAX_BYTES) -> None:
        self._cache_root = cache_root
        self._run = runner
        self._max_bytes = max_bytes
        # One ffmpeg at a time: a browser fires several Range requests at once, and the box is busy with the
        # appliance. The second request waits, then finds the first one's result in the cache.
        self._jobs = asyncio.Semaphore(1)

    async def probe(self, path: Path) -> Probe:
        code, out = await self._run(
            ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", str(path)], PROBE_TIMEOUT_SECONDS)
        if code != 0:
            raise MediaError("This recording can't be read.")
        return parse_probe(out)

    async def prepare(self, offer: Offer, variant: Variant) -> Prepared:
        """The file to serve for `variant`. Raises MediaError when it can't be made."""
        if variant not in offer.variants:
            raise MediaError(f"'{variant.value}' isn't available for this source.")
        if variant is Variant.ORIGINAL:
            return Prepared(offer.path, offer.mime)
        if variant is Variant.TEXT:
            return await self._cached(offer, variant, ".txt", "text/plain; charset=utf-8", self._write_text)
        if variant is Variant.EXCERPT:
            if offer.kind is SourceKind.VIDEO:
                return await self._cached(offer, variant, ".mp4", "video/mp4", self._ffmpeg(self._video_cut))
            return await self._cached(offer, variant, ".m4a", "audio/mp4", self._ffmpeg(self._audio_cut))
        # MEDIA
        if offer.kind is SourceKind.PDF:
            return Prepared(offer.path, offer.mime)
        if offer.kind is SourceKind.IMAGE:
            if offer.path.suffix.lower() in _PLAYABLE_IMAGES:
                return Prepared(offer.path, offer.mime)
            return await self._cached(offer, variant, ".png", "image/png", self._ffmpeg(self._image_png))
        if browser_playable(offer.kind, offer.path.suffix, await self.probe(offer.path)):
            # An .m4a/.ogg stays its own type; a playable video is mp4 or webm, which the table already names.
            return Prepared(offer.path, offer.mime)
        if offer.kind is SourceKind.VIDEO:
            return await self._cached(offer, variant, ".mp4", "video/mp4", self._ffmpeg(self._video_h264))
        return await self._cached(offer, variant, ".m4a", "audio/mp4", self._ffmpeg(self._audio_aac))

    # ---------- the cache ----------

    def cache_path(self, offer: Offer, variant: Variant, suffix: str) -> Path:
        """Named by the agent alone: the resolved source, its size and mtime, and (for a cut) the window."""
        stat = offer.path.stat()
        parts = [str(offer.path), str(stat.st_size), str(stat.st_mtime_ns), variant.value]
        if variant is Variant.EXCERPT:
            parts += [f"{t:.3f}" for t in offer.excerpt_window]
        return self._cache_root / (hashlib.sha256("|".join(parts).encode()).hexdigest() + suffix)

    async def _cached(self, offer: Offer, variant: Variant, suffix: str, mime: str,
                      make: Callable[[Offer, Path], Awaitable[None]]) -> Prepared:
        target = self.cache_path(offer, variant, suffix)
        if target.exists():
            return Prepared(target, mime)
        async with self._jobs:
            if target.exists():
                return Prepared(target, mime)
            self._cache_root.mkdir(parents=True, exist_ok=True)
            # Written beside the target and renamed at the end, so a half-made file is never served.
            partial = target.with_name(target.stem + ".part" + suffix)
            try:
                await make(offer, partial)
                partial.replace(target)
            finally:
                partial.unlink(missing_ok=True)
            await asyncio.to_thread(self.prune, keep=target)
        return Prepared(target, mime)

    def prune(self, keep: Path) -> None:
        """Deletes the oldest cached files until the cache fits its budget again, never the one just made."""
        files = sorted((p for p in self._cache_root.iterdir() if p.is_file() and p != keep),
                       key=lambda p: p.stat().st_mtime)
        total = sum(p.stat().st_size for p in files) + (keep.stat().st_size if keep.exists() else 0)
        for path in files:
            if total <= self._max_bytes:
                break
            total -= path.stat().st_size
            path.unlink(missing_ok=True)

    async def _write_text(self, offer: Offer, out: Path) -> None:
        data = await asyncio.to_thread(offer.path.read_bytes)
        out.write_text(await asyncio.to_thread(decode_text, data), encoding="utf-8")

    def _ffmpeg(self, arguments: Callable[[Offer, Path], List[str]]):
        async def make(offer: Offer, out: Path) -> None:
            command = ["nice", "-n", "10", "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                       *arguments(offer, out)]
            code, _ = await self._run(command, FFMPEG_TIMEOUT_SECONDS)
            if code != 0 or not out.exists():
                raise MediaError("The appliance couldn't prepare this source for the browser.")
        return make

    @staticmethod
    def _audio_cut(offer: Offer, out: Path) -> List[str]:
        start, end = offer.excerpt_window
        return ["-ss", f"{start:.3f}", "-i", str(offer.path), "-t", f"{end - start:.3f}", "-vn", *_AAC,
                "-movflags", "+faststart", "-f", "mp4", str(out)]

    @staticmethod
    def _video_cut(offer: Offer, out: Path) -> List[str]:
        # Re-encoded, not copied: a stream copy can only cut on keyframes, seconds away from the cited word.
        start, end = offer.excerpt_window
        return ["-ss", f"{start:.3f}", "-i", str(offer.path), "-t", f"{end - start:.3f}",
                "-map", "0:v:0?", "-map", "0:a:0?", *_H264, *_AAC, "-movflags", "+faststart", "-f", "mp4", str(out)]

    @staticmethod
    def _video_h264(offer: Offer, out: Path) -> List[str]:
        return ["-i", str(offer.path), "-map", "0:v:0?", "-map", "0:a:0?", *_H264, *_AAC,
                "-movflags", "+faststart", "-f", "mp4", str(out)]

    @staticmethod
    def _audio_aac(offer: Offer, out: Path) -> List[str]:
        return ["-i", str(offer.path), "-vn", *_AAC, "-movflags", "+faststart", "-f", "mp4", str(out)]

    @staticmethod
    def _image_png(offer: Offer, out: Path) -> List[str]:
        return ["-i", str(offer.path), "-frames:v", "1", "-f", "image2", "-c:v", "png", str(out)]
