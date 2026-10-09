import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.features.media_compaction.domain.entities import MediaProbe
from app.features.media_compaction.domain.interfaces import IMediaTranscoder
from app.features.media_compaction.utils.error_mapper import map_ffmpeg_error

logger = logging.getLogger(__name__)

# Speech-quality stereo beside the picture; an audio-only recording gets the configured bitrate instead
VIDEO_AUDIO_BITRATE = "96k"


def parse_probe(payload: Dict[str, Any]) -> MediaProbe:
    """ffprobe's JSON → the first real video and audio streams. Cover art (attached_pic) is not a video."""
    video: Optional[Dict[str, Any]] = None
    audio: Optional[Dict[str, Any]] = None
    for stream in payload.get("streams") or []:
        kind = stream.get("codec_type")
        if kind == "video" and video is None and not (stream.get("disposition") or {}).get("attached_pic"):
            video = stream
        elif kind == "audio" and audio is None:
            audio = stream
    fmt = payload.get("format") or {}
    file_bit_rate = _as_int(fmt.get("bit_rate"))
    audio_bit_rate = _as_int(audio.get("bit_rate")) if audio else None
    if audio and video is None and audio_bit_rate is None:
        audio_bit_rate = file_bit_rate
    return MediaProbe(
        video_codec=video.get("codec_name") if video else None,
        width=_as_int(video.get("width")) if video else None,
        height=_as_int(video.get("height")) if video else None,
        audio_codec=audio.get("codec_name") if audio else None,
        audio_bit_rate=audio_bit_rate,
        bit_rate=file_bit_rate,
        duration_sec=_as_float(fmt.get("duration")),
    )


def scale_filter(max_height: int) -> str:
    """
    Caps the SHORT side at `max_height`, so a portrait phone video (autorotated by ffmpeg) is treated like a
    landscape one, and never upscales. Both sides stay even, as yuv420p H.264 requires.
    """
    short = f"min({max_height},trunc(ih/2)*2)"
    short_w = f"min({max_height},trunc(iw/2)*2)"
    return f"scale=w='if(gte(iw,ih),-2,{short_w})':h='if(gte(iw,ih),{short},-2)'"


class FFmpegTranscoder(IMediaTranscoder):
    def __init__(self, timeout_seconds: int = 6 * 3600):
        self._timeout = timeout_seconds

    def probe(self, path: str | Path) -> MediaProbe:
        try:
            result = self._run([
                "ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format",
                os.fspath(path),
            ])
            return parse_probe(json.loads(result.stdout or "{}"))
        except Exception as e:
            raise map_ffmpeg_error(e) from e

    def compact_video(self, source_path: str | Path, dest_path: str | Path, max_height: int, crf: int) -> None:
        self._transcode([
            "-i", os.fspath(source_path),
            "-map", "0:v:0", "-map", "0:a:0?",
            "-vf", scale_filter(max_height),
            "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", VIDEO_AUDIO_BITRATE,
            # The browser can start playing (and seek to a citation) before the whole file has arrived
            "-movflags", "+faststart",
            "-f", "mp4",
        ], dest_path)

    def compact_audio(self, source_path: str | Path, dest_path: str | Path, bitrate_kbps: int) -> None:
        self._transcode([
            "-i", os.fspath(source_path),
            "-map", "0:a:0", "-vn",
            "-c:a", "aac", "-b:a", f"{bitrate_kbps}k",
            "-movflags", "+faststart",
            "-f", "mp4",
        ], dest_path)

    def _transcode(self, args: List[str], dest_path: str | Path) -> None:
        dest = Path(dest_path)
        # Niced: on the hosted box this runs beside the live API, which must keep answering
        command = ["nice", "-n", "10", "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", *args, "-y",
                   os.fspath(dest)]
        try:
            self._run(command)
        except Exception as e:
            dest.unlink(missing_ok=True)
            raise map_ffmpeg_error(e) from e

    def _run(self, command: List[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            command,
            # FFmpeg reads the terminal for interactive keys and switches it to raw mode
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            timeout=self._timeout,
        )


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value) if value not in (None, "", "N/A") else None
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value) if value not in (None, "", "N/A") else None
    except (TypeError, ValueError):
        return None
