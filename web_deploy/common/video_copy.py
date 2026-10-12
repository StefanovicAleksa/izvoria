"""
One video's 720p copy: what the box ingests and the vault keeps, made before the recording leaves the machine it is
on. Used by the VPS's preparer (web_deploy/gateway/prepare.py) and by the firm's own uploader (web_deploy/uploader),
which makes the copy on the firm's computer so only the copy crosses the internet.

Standard library only, with ffmpeg/ffprobe on the PATH. The encode is the box compaction's own
(`compact_video_args`), and a copy the box's rule calls compact (`is_compact_video`) is never re-encoded there.
"""

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from app.features.media_compaction.utils.ffmpeg_args import compact_video_args, is_compact_video
from app.features.storage.utils.upload_record import UploadRecord

HASH_CHUNK_BYTES = 8 * 1024 * 1024
# How far the copy's length may drift from the original's (as the box's compaction allows): chunk timestamps index
# into the kept file, and the copy is that file
DURATION_TOLERANCE_SEC = 0.5
DURATION_TOLERANCE_RATIO = 0.002

Runner = Callable[..., subprocess.CompletedProcess]


class CopyFailed(RuntimeError):
    pass


@dataclass(frozen=True)
class CopySettings:
    max_height: int = 720
    crf: int = 26
    # x264's speed: the CPU making the copies is what a batch waits on. "faster" keeps a talk under the box's
    # 3 Mbit/s already-compact line at 720p; slower presets make smaller files, more slowly.
    preset: str = "faster"
    # Above the vault's 96k: this audio is what Whisper transcribes, not just what is played back
    audio_bitrate: str = "128k"
    timeout_seconds: int = 12 * 3600
    # Lower priority than everything else on the machine (the gateway's requests, the user's own work)
    nice: bool = True


@dataclass(frozen=True)
class Probe:
    video_codec: Optional[str]
    audio_codec: Optional[str]
    width: Optional[int]
    height: Optional[int]
    bit_rate: Optional[int]
    duration_sec: Optional[float]


def parse_probe(payload: dict) -> Probe:
    """ffprobe's JSON: the first real video stream (cover art is not one), the first audio stream, the container."""
    video = audio = None
    for stream in payload.get("streams") or []:
        if stream.get("codec_type") == "video" and video is None \
                and not (stream.get("disposition") or {}).get("attached_pic"):
            video = stream
        elif stream.get("codec_type") == "audio" and audio is None:
            audio = stream
    fmt = payload.get("format") or {}
    return Probe(
        video_codec=video.get("codec_name") if video else None,
        audio_codec=audio.get("codec_name") if audio else None,
        width=_int(video.get("width")) if video else None,
        height=_int(video.get("height")) if video else None,
        bit_rate=_int(fmt.get("bit_rate")),
        duration_sec=_float(fmt.get("duration")),
    )


def sha256_of(path: Path) -> str:
    """The hash ingestion dedups a source on (app/features/storage/utils/hashing.py: sha256, hex)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def probe(path: Path, run: Runner = subprocess.run, timeout: int = 600) -> Probe:
    completed = _call(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format",
                       os.fspath(path)], run, timeout)
    return parse_probe(json.loads(completed.stdout or "{}"))


def needs_copy(source: Path, info: Probe, settings: CopySettings) -> bool:
    """Whether a smaller copy is worth making: a picture, not already a compact MP4."""
    if info.video_codec is None:
        return False
    return not is_compact_video(source.suffix, info.video_codec, info.audio_codec, info.width, info.height,
                                info.bit_rate, settings.max_height)


def make_copy(source: Path, dest: Path, settings: CopySettings, run: Runner = subprocess.run) -> Optional[UploadRecord]:
    """
    Writes the 720p copy of `source` to `dest` and returns what the original was (its hash and name), or None when
    no copy is worth making. Raises CopyFailed, leaving nothing at `dest`, when ffmpeg fails or the copy's length
    drifted: keep the original then.
    """
    original = probe(source, run)
    if not needs_copy(source, original, settings):
        return None
    record = UploadRecord(sha256=sha256_of(source), original_name=source.name)
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
               *compact_video_args(source, settings.max_height, settings.crf, settings.preset, settings.audio_bitrate),
               "-y", os.fspath(dest)]
    if settings.nice and os.name == "posix":
        command = ["nice", "-n", "10", *command]
    try:
        _call(command, run, settings.timeout_seconds)
        refusal = length_refusal(original, probe(dest, run))
        if refusal:
            raise CopyFailed(refusal)
    except Exception as exc:
        Path(dest).unlink(missing_ok=True)
        raise exc if isinstance(exc, CopyFailed) else CopyFailed(str(exc)) from exc
    return record


def length_refusal(original: Probe, copy: Probe) -> Optional[str]:
    if original.duration_sec is None or copy.duration_sec is None:
        return "its length could not be read, so the copy's timeline could not be checked"
    tolerance = max(DURATION_TOLERANCE_SEC, original.duration_sec * DURATION_TOLERANCE_RATIO)
    if abs(original.duration_sec - copy.duration_sec) > tolerance:
        return f"the copy runs {copy.duration_sec:.2f}s against the original's {original.duration_sec:.2f}s"
    return None


def copy_name(source: Path, taken: set) -> str:
    """`<stem>.mp4`, the upload's own name when it was an MP4; numbered if another file of the batch has it."""
    name = f"{source.stem}.mp4"
    if name == source.name or name not in taken:
        return name
    counter = 2
    while f"{source.stem}-{counter}.mp4" in taken:
        counter += 1
    return f"{source.stem}-{counter}.mp4"


def _call(command: List[str], run: Runner, timeout: int) -> subprocess.CompletedProcess:
    completed = run(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                    timeout=timeout)
    if completed.returncode != 0:
        tool = "ffprobe" if "ffprobe" in command[:1] else "ffmpeg"
        raise CopyFailed(f"{tool} failed: {(completed.stderr or '').strip()[-500:]}")
    return completed


def _int(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
