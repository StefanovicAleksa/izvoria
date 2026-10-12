# Path: app/features/media_compaction/utils/ffmpeg_args.py
"""
What a compact video is, and the ffmpeg arguments that make one. Standard library only, and the package's
__init__ files are empty, so the hosted gateway (which has no appliance dependencies) imports this module too: its
video preparer makes the 720p copy before the GPU box ever sees a recording (web_deploy/storage/prepare.py), and the
box's compaction must then recognise that copy as already compact rather than re-encode it.
"""

import os
from pathlib import Path
from typing import List, Optional

# Speech-quality stereo beside the picture; an audio-only recording gets the configured bitrate instead
VIDEO_AUDIO_BITRATE = "96k"
# A browser-playable MP4 at or under the target height is left alone up to this bitrate (a YouTube download is)
COMPACT_VIDEO_MAX_BIT_RATE = 3_000_000


def scale_filter(max_height: int) -> str:
    """
    Caps the SHORT side at `max_height`, so a portrait phone video (autorotated by ffmpeg) is treated like a
    landscape one, and never upscales. Both sides stay even, as yuv420p H.264 requires.
    """
    short = f"min({max_height},trunc(ih/2)*2)"
    short_w = f"min({max_height},trunc(iw/2)*2)"
    return f"scale=w='if(gte(iw,ih),-2,{short_w})':h='if(gte(iw,ih),{short},-2)'"


def compact_video_args(source_path: str | Path, max_height: int, crf: int, preset: str = "medium",
                       audio_bitrate: str = VIDEO_AUDIO_BITRATE) -> List[str]:
    """ffmpeg's arguments between the binary and the output path: H.264 + AAC MP4, playable as it arrives."""
    return [
        "-i", os.fspath(source_path),
        "-map", "0:v:0", "-map", "0:a:0?",
        "-vf", scale_filter(max_height),
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", audio_bitrate,
        # The browser can start playing (and seek to a citation) before the whole file has arrived
        "-movflags", "+faststart",
        "-f", "mp4",
    ]


def is_compact_video(suffix: str, video_codec: Optional[str], audio_codec: Optional[str], width: Optional[int],
                     height: Optional[int], bit_rate: Optional[int], max_height: int) -> bool:
    """An MP4 every browser plays, no taller than `max_height` on its short side, at a modest bitrate."""
    short_side = min(width or 0, height or 0)
    return (suffix.lower() == ".mp4" and video_codec == "h264" and audio_codec in (None, "aac")
            and 0 < short_side <= max_height
            and bit_rate is not None and bit_rate <= COMPACT_VIDEO_MAX_BIT_RATE)
