import os
import shutil
import subprocess
from pathlib import Path

import pytest

from app.features.media_compaction.data.ffmpeg_transcoder import FFmpegTranscoder
from app.features.media_compaction.domain.entities import CompactionStatus
from app.features.media_compaction.service import compaction_api as compaction_module
from app.features.media_compaction.service.compaction_api import CompactionAPI
from app.features.storage.data.repositories import StorageRepository
from app.features.storage.domain.entities import SourceType
from app.features.storage.service.storage_api import StorageAPI

RESOURCES = Path(__file__).resolve().parents[2] / "resources"

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def require_ffmpeg():
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        pytest.skip("ffmpeg/ffprobe are not installed on this system.")


@pytest.fixture(autouse=True)
def web_settings(monkeypatch):
    monkeypatch.setattr(compaction_module.settings, "COMPACT_VIDEO_MAX_HEIGHT", 720)
    monkeypatch.setattr(compaction_module.settings, "COMPACT_VIDEO_CRF", 26)
    monkeypatch.setattr(compaction_module.settings, "COMPACT_AUDIO_BITRATE_KBPS", 64)
    monkeypatch.setattr(compaction_module.settings, "COMPACT_MIN_SAVING", 0.2)


@pytest.fixture
def storage(db_manager, tmp_path):
    api = StorageAPI(StorageRepository(db_manager))
    api.file_manager.base_path = tmp_path / "vault"
    api.file_manager.base_path.mkdir()
    return api


@pytest.fixture
def compaction(storage):
    return CompactionAPI(transcoder=FFmpegTranscoder(timeout_seconds=600), storage_api=storage)


def _camera_recording(path: Path, seconds: int = 8) -> None:
    """A heavy 1080p recording, as a phone or camera writes it: high bitrate, in a container browsers won't play."""
    subprocess.run([
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", f"testsrc2=duration={seconds}:size=1920x1080:rate=30",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}:sample_rate=48000",
        "-c:v", "libx264", "-preset", "ultrafast", "-b:v", "12M", "-c:a", "aac", "-b:a", "256k",
        os.fspath(path),
    ], check=True)


class TestMediaCompactionIntegration:
    def test_a_1080p_recording_is_replaced_by_a_720p_copy_on_the_same_timeline(self, storage, compaction,
                                                                             tmp_path):
        upload = tmp_path / "upload" / "deposition.mkv"
        upload.parent.mkdir()
        _camera_recording(upload)
        source = storage.ingest_local_file(os.fspath(upload), SourceType.VIDEO)
        original_path = storage.get_file_path(source.file_id)
        before = original_path.stat().st_size
        transcoder = FFmpegTranscoder()
        original_probe = transcoder.probe(original_path)

        outcome = compaction.compact_source(source.id)

        assert outcome.status is CompactionStatus.REPLACED, outcome.reason
        compact_path = storage.get_file_path(source.file_id)
        assert compact_path.suffix == ".mp4"
        assert not original_path.exists()
        record = storage.get_file(source.file_id)
        assert (record.file_size_bytes, record.mime_type) == (compact_path.stat().st_size, "video/mp4")
        assert record.file_size_bytes < before * 0.8
        probe = transcoder.probe(compact_path)
        assert (probe.video_codec, probe.width, probe.height, probe.audio_codec) == ("h264", 1280, 720, "aac")
        assert abs(probe.duration_sec - original_probe.duration_sec) < 0.5
        # The moov atom comes first, so a browser can seek to a citation before the whole file has arrived
        assert b"moov" in compact_path.read_bytes()[:4096]
        # Nothing but the compact copy is left in the vault
        assert [p.name for p in compact_path.parent.iterdir()] == [compact_path.name]
        # The same upload still dedups to the same source
        assert storage.ingest_local_file(os.fspath(upload), SourceType.VIDEO).id == source.id

    def test_a_wav_recording_becomes_a_small_m4a(self, storage, compaction, tmp_path):
        upload = tmp_path / "call.wav"
        shutil.copy(RESOURCES / "test_speech.wav", upload)
        source = storage.ingest_local_file(os.fspath(upload), SourceType.AUDIO)
        before = storage.get_file(source.file_id).file_size_bytes

        outcome = compaction.compact_source(source.id)

        assert outcome.status is CompactionStatus.REPLACED, outcome.reason
        compact_path = storage.get_file_path(source.file_id)
        assert compact_path.suffix == ".m4a"
        assert storage.get_file(source.file_id).file_size_bytes < before * 0.2
        assert FFmpegTranscoder().probe(compact_path).audio_codec == "aac"

    def test_a_lean_browser_ready_video_is_left_alone(self, storage, compaction, tmp_path):
        upload = tmp_path / "lecture.mp4"
        shutil.copy(RESOURCES / "test_video.mp4", upload)  # 1280x534 H.264/AAC at ~1.6 Mbit/s
        source = storage.ingest_local_file(os.fspath(upload), SourceType.VIDEO)
        path = storage.get_file_path(source.file_id)

        outcome = compaction.compact_source(source.id)

        assert outcome.status is CompactionStatus.ALREADY_COMPACT
        assert storage.get_file_path(source.file_id) == path
