import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from app.features.storage.utils.upload_record import UploadRecord, read_all, write_record
from web_deploy.common.video_copy import parse_probe
from web_deploy.gateway.prepare import PrepareSettings, VideoPreparer
from web_deploy.ingest_worker.queue import IngestMode, IngestQueue, JobStatus

RAW = {"streams": [{"codec_type": "video", "codec_name": "hevc", "width": 3840, "height": 2160},
                   {"codec_type": "audio", "codec_name": "pcm_s16le"}],
       "format": {"duration": "600.0", "bit_rate": "90000000"}}
COPY = {"streams": [{"codec_type": "video", "codec_name": "h264", "width": 1280, "height": 720},
                    {"codec_type": "audio", "codec_name": "aac"}],
        "format": {"duration": "600.1", "bit_rate": "1800000"}}


class FakeFfmpeg:
    """ffprobe answers per file; ffmpeg writes a small 'copy' and records its arguments."""

    def __init__(self, probes=None, fail=False):
        self.probes = probes or {}
        self.fail = fail
        self.encoded = []

    def __call__(self, command, **kwargs):
        if command[0] == "ffprobe":
            path = Path(command[-1])
            payload = self.probes.get(path.name, COPY if path.name.startswith(".preparing-") else RAW)
            return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")
        assert command[:3] == ["nice", "-n", "10"] and "ffmpeg" in command
        if self.fail:
            return subprocess.CompletedProcess(command, 1, "", "Invalid data found when processing input")
        self.encoded.append(command)
        Path(command[-1]).write_bytes(b"720p copy")
        return subprocess.CompletedProcess(command, 0, "", "")


@pytest.fixture
def queue():
    return IngestQueue.connect("sqlite://")


def _batch(tmp_path, files):
    folder = tmp_path / "legal-demo" / "uploads" / "u1" / "20261012-120000-abcdef"
    folder.mkdir(parents=True)
    for name, content in files.items():
        (folder / name).write_bytes(content)
    return folder


def _job(queue, folder, mode=IngestMode.VIDEO):
    return queue.enqueue("legal-demo", "u1", mode, "en", str(folder), len(list(folder.iterdir())), prepare=True)


class TestVideoPreparer:
    def test_the_box_gets_a_720p_copy_known_by_the_uploads_hash_and_name(self, tmp_path, queue):
        folder = _batch(tmp_path, {"Deposition 4K.mov": b"raw footage", "exhibit.pdf": b"%PDF"})
        job = _job(queue, folder)
        ffmpeg = FakeFfmpeg()

        prepared = VideoPreparer(queue, PrepareSettings(enabled=True), run=ffmpeg).run_once()

        assert prepared.id == job.id
        assert sorted(p.name for p in folder.iterdir()) == [".upload-record.json", "Deposition 4K.mp4", "exhibit.pdf"]
        assert read_all(folder) == {"Deposition 4K.mp4": UploadRecord(hashlib.sha256(b"raw footage").hexdigest(),
                                                                      "Deposition 4K.mov")}
        assert queue.get(job.id).status is JobStatus.QUEUED
        command = ffmpeg.encoded[0]
        assert command[command.index("-preset") + 1] == "faster"
        assert command[command.index("-b:a") + 1] == "128k"

    def test_an_already_compact_mp4_and_audio_are_left_alone(self, tmp_path, queue):
        folder = _batch(tmp_path, {"youtube.mp4": b"compact", "call.m4a": b"audio"})
        _job(queue, folder)
        ffmpeg = FakeFfmpeg(probes={"youtube.mp4": COPY,
                                    "call.m4a": {"streams": [{"codec_type": "audio", "codec_name": "aac"}]}})

        VideoPreparer(queue, PrepareSettings(enabled=True), run=ffmpeg).run_once()

        assert ffmpeg.encoded == []
        assert (folder / "youtube.mp4").read_bytes() == b"compact"
        assert read_all(folder) == {}

    def test_a_large_mp4_is_replaced_in_place(self, tmp_path, queue):
        folder = _batch(tmp_path, {"talk.mp4": b"4k mp4"})
        _job(queue, folder)

        VideoPreparer(queue, PrepareSettings(enabled=True), run=FakeFfmpeg()).run_once()

        assert (folder / "talk.mp4").read_bytes() == b"720p copy"
        assert read_all(folder)["talk.mp4"].original_name == "talk.mp4"

    def test_a_name_another_file_has_is_numbered(self, tmp_path, queue):
        folder = _batch(tmp_path, {"talk.mov": b"mov", "talk.mp4": b"other"})
        _job(queue, folder)
        ffmpeg = FakeFfmpeg(probes={"talk.mp4": COPY})

        VideoPreparer(queue, PrepareSettings(enabled=True), run=ffmpeg).run_once()

        assert (folder / "talk.mp4").read_bytes() == b"other"
        assert read_all(folder)["talk-2.mp4"].original_name == "talk.mov"

    @pytest.mark.parametrize("ffmpeg", [FakeFfmpeg(fail=True),
                                        FakeFfmpeg(probes={".preparing-talk.mp4": {**COPY, "format": {
                                            "duration": "540.0", "bit_rate": "1800000"}}})])
    def test_a_failed_or_shortened_copy_keeps_the_original_and_the_job_goes_on(self, tmp_path, queue, ffmpeg):
        folder = _batch(tmp_path, {"talk.mov": b"raw"})
        job = _job(queue, folder)

        VideoPreparer(queue, PrepareSettings(enabled=True), run=ffmpeg).run_once()

        assert sorted(p.name for p in folder.iterdir()) == ["talk.mov"]
        assert queue.get(job.id).status is JobStatus.QUEUED

    def test_a_cancelled_job_is_not_handed_to_the_box(self, tmp_path, queue):
        folder = _batch(tmp_path, {"a.mov": b"a", "b.mov": b"b"})
        job = _job(queue, folder)
        queue.cancel(job.id, "legal-demo")

        assert VideoPreparer(queue, PrepareSettings(enabled=True), run=FakeFfmpeg()).run_once() is None
        assert queue.get(job.id).status is JobStatus.CANCELLED

    def test_a_restart_finishes_a_half_done_swap_and_drops_a_partial_copy(self, tmp_path, queue):
        folder = _batch(tmp_path, {"a.mov": b"a raw", "a.mp4": b"a copy", ".preparing-b.mp4": b"partial",
                                   "b.mov": b"b raw"})
        write_record(folder, {"a.mp4": UploadRecord("1" * 64, "a.mov")})
        _job(queue, folder)
        ffmpeg = FakeFfmpeg(probes={"a.mp4": COPY})

        VideoPreparer(queue, PrepareSettings(enabled=True), run=ffmpeg).run_once()

        assert sorted(p.name for p in folder.iterdir()) == [".upload-record.json", "a.mp4", "b.mp4"]
        assert [Path(c[-1]).name for c in ffmpeg.encoded] == [".preparing-b.mp4"]

    def test_nothing_to_prepare(self, queue):
        assert VideoPreparer(queue, PrepareSettings(enabled=True), run=FakeFfmpeg()).run_once() is None


class TestSettings:
    def test_off_unless_asked_for(self):
        assert PrepareSettings.from_env({}).enabled is False
        on = PrepareSettings.from_env({"PREPARE_VIDEOS": "true", "PREPARE_PRESET": "veryfast"})
        assert (on.enabled, on.copy.preset, on.copy.max_height) == (True, "veryfast", 720)


class TestProbe:
    def test_cover_art_is_not_a_video(self):
        probe = parse_probe({"streams": [{"codec_type": "video", "codec_name": "mjpeg", "disposition": {"attached_pic": 1}},
                                         {"codec_type": "audio", "codec_name": "mp3"}], "format": {}})
        assert (probe.video_codec, probe.audio_codec) == (None, "mp3")
