# Path: tests/features/media_compaction/utils/test_ffmpeg_args.py

import pytest

from app.features.media_compaction.utils.ffmpeg_args import (
    COMPACT_VIDEO_MAX_BIT_RATE, compact_video_args, is_compact_video, scale_filter,
)


class TestCompactVideoArgs:
    def test_an_h264_aac_mp4_capped_on_its_short_side(self):
        args = compact_video_args("/in/talk.mov", 720, 26)

        assert args[:2] == ["-i", "/in/talk.mov"]
        assert args[args.index("-vf") + 1] == scale_filter(720)
        assert args[args.index("-c:v") + 1] == "libx264" and args[args.index("-crf") + 1] == "26"
        assert args[args.index("-preset") + 1] == "medium"
        assert args[-2:] == ["-f", "mp4"] and "+faststart" in args

    def test_the_preset_and_audio_bitrate_can_be_chosen(self):
        args = compact_video_args("/in/a.mp4", 720, 26, preset="veryfast", audio_bitrate="128k")
        assert args[args.index("-preset") + 1] == "veryfast"
        assert args[args.index("-b:a") + 1] == "128k"


class TestIsCompactVideo:
    @pytest.mark.parametrize("suffix, video, audio, w, h, rate, expected", [
        (".mp4", "h264", "aac", 1280, 720, 2_000_000, True),
        (".mp4", "h264", None, 720, 1280, 2_000_000, True),          # portrait: the short side counts
        (".mov", "h264", "aac", 1280, 720, 2_000_000, False),        # not what every browser plays
        (".mp4", "hevc", "aac", 1280, 720, 2_000_000, False),
        (".mp4", "h264", "aac", 3840, 2160, 2_000_000, False),
        (".mp4", "h264", "aac", 1280, 720, COMPACT_VIDEO_MAX_BIT_RATE + 1, False),
        (".mp4", "h264", "aac", 1280, 720, None, False),
    ])
    def test_the_rule(self, suffix, video, audio, w, h, rate, expected):
        assert is_compact_video(suffix, video, audio, w, h, rate, 720) is expected
