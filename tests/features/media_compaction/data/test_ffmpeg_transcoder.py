import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.features.media_compaction import exceptions as exc
from app.features.media_compaction.data.ffmpeg_transcoder import FFmpegTranscoder, parse_probe, scale_filter
from app.features.media_compaction.domain.entities import MediaProbe

RUN = "app.features.media_compaction.data.ffmpeg_transcoder.subprocess.run"


class TestParseProbe:
    def test_reads_the_first_video_and_audio_streams_and_the_length(self):
        payload = {
            "streams": [
                {"codec_type": "video", "codec_name": "hevc", "width": 1920, "height": 1080},
                {"codec_type": "audio", "codec_name": "aac", "bit_rate": "128000"},
                {"codec_type": "audio", "codec_name": "ac3", "bit_rate": "384000"},
            ],
            "format": {"duration": "3600.5", "bit_rate": "8000000"},
        }

        assert parse_probe(payload) == MediaProbe(video_codec="hevc", width=1920, height=1080, audio_codec="aac",
                                                  audio_bit_rate=128000, bit_rate=8000000, duration_sec=3600.5)

    def test_cover_art_is_not_a_video_stream(self):
        payload = {"streams": [
            {"codec_type": "video", "codec_name": "mjpeg", "width": 600, "height": 600,
             "disposition": {"attached_pic": 1}},
            {"codec_type": "audio", "codec_name": "mp3"},
        ], "format": {"duration": "60", "bit_rate": "192000"}}

        probe = parse_probe(payload)

        assert probe.video_codec is None
        # An audio-only file whose stream doesn't state its rate takes the file's
        assert probe.audio_bit_rate == 192000

    def test_a_video_files_rate_is_not_taken_for_its_audio(self):
        payload = {"streams": [{"codec_type": "video", "codec_name": "h264", "width": 1280, "height": 720},
                               {"codec_type": "audio", "codec_name": "aac"}],
                   "format": {"bit_rate": "2000000"}}

        assert parse_probe(payload).audio_bit_rate is None

    def test_missing_and_na_values_become_none(self):
        probe = parse_probe({"streams": [], "format": {"duration": "N/A"}})

        assert probe == MediaProbe()


class TestScaleFilter:
    def test_caps_the_short_side_and_keeps_both_sides_even(self):
        assert scale_filter(720) == ("scale=w='if(gte(iw,ih),-2,min(720,trunc(iw/2)*2))'"
                                     ":h='if(gte(iw,ih),min(720,trunc(ih/2)*2),-2)'")


class TestFFmpegTranscoder:
    def test_probe_runs_ffprobe_for_streams_and_format(self):
        result = MagicMock(stdout=json.dumps({"streams": [{"codec_type": "audio", "codec_name": "aac"}],
                                              "format": {"duration": "5"}}))
        with patch(RUN, return_value=result) as run:
            probe = FFmpegTranscoder(timeout_seconds=30).probe(Path("/vault/a.m4a"))

        assert probe.audio_codec == "aac"
        assert probe.duration_sec == 5.0
        run.assert_called_once_with(
            ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", "/vault/a.m4a"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True,
            timeout=30)

    def test_probe_failure_is_typed(self):
        with patch(RUN, side_effect=subprocess.CalledProcessError(1, "ffprobe", stderr="Invalid data")):
            with pytest.raises(exc.FFmpegExecutionError, match="Invalid data"):
                FFmpegTranscoder().probe("/vault/broken.mp4")

    def test_compact_video_writes_a_capped_h264_aac_mp4(self):
        with patch(RUN) as run:
            FFmpegTranscoder(timeout_seconds=99).compact_video("/vault/a.mkv", "/vault/.c.mp4", 720, 26)

        command = run.call_args.args[0]
        assert command == [
            "nice", "-n", "10", "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-i", "/vault/a.mkv", "-map", "0:v:0", "-map", "0:a:0?", "-vf", scale_filter(720),
            "-c:v", "libx264", "-preset", "medium", "-crf", "26", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", "-f", "mp4", "-y", "/vault/.c.mp4",
        ]
        assert run.call_args.kwargs["timeout"] == 99
        assert run.call_args.kwargs["stdin"] is subprocess.DEVNULL

    def test_compact_audio_writes_an_aac_m4a_at_the_bitrate(self):
        with patch(RUN) as run:
            FFmpegTranscoder().compact_audio("/vault/a.wav", "/vault/.c.m4a", 64)

        assert run.call_args.args[0] == [
            "nice", "-n", "10", "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-i", "/vault/a.wav", "-map", "0:a:0", "-vn", "-c:a", "aac", "-b:a", "64k",
            "-movflags", "+faststart", "-f", "mp4", "-y", "/vault/.c.m4a",
        ]

    def test_a_failed_transcode_removes_its_partial_output(self, tmp_path):
        partial = tmp_path / ".c.mp4"
        partial.write_bytes(b"half")
        with patch(RUN, side_effect=subprocess.TimeoutExpired(cmd="ffmpeg", timeout=5)):
            with pytest.raises(exc.FFmpegTimeoutError):
                FFmpegTranscoder().compact_video(tmp_path / "a.mkv", partial, 720, 26)

        assert not partial.exists()
