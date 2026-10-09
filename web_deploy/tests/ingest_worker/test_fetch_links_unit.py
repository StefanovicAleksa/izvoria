import json
from pathlib import Path

from web_deploy.common import youtube
from web_deploy.common.youtube import Video
from web_deploy.ingest_worker.fetch_links import _js_runtimes, fetch
from web_deploy.ingest_worker.queue import IngestMode, IngestQueue, JobOrigin, JobStatus

VIDEOS = [Video("video000001", "Lecture 1"), Video("video000002", "Lecture 2"), Video("video000003", "Lecture 3")]


def _job(tmp_path, videos=VIDEOS):
    queue = IngestQueue.connect("sqlite://")
    youtube.write_manifest(tmp_path, list(videos))
    job = queue.enqueue("legal-demo", "u1", IngestMode.VIDEO, "en", str(tmp_path), len(videos),
                        origin=JobOrigin.YOUTUBE)
    queue.claim_next()
    return queue, job


class FakeDownloader:
    def __init__(self, failing=()):
        self.failing = set(failing)
        self.calls = []

    def __call__(self, video, folder, on_fraction):
        self.calls.append(video.id)
        on_fraction(0.5)
        if video.id in self.failing:
            raise RuntimeError(f"ERROR: [youtube] {video.id}: Video unavailable")
        (Path(folder) / video.file_name).write_bytes(b"mp4")


class TestFetch:
    def test_every_video_is_downloaded_under_its_title_and_the_progress_is_recorded(self, tmp_path):
        queue, job = _job(tmp_path)
        clock = iter(range(0, 1000, 10))

        code = fetch(job.id, queue, FakeDownloader(), clock=lambda: next(clock), wall=lambda: 5.0)

        assert code == 0
        assert sorted(p.name for p in tmp_path.glob("*.mp4")) == [v.file_name for v in VIDEOS]
        assert queue.get(job.id).progress == {"phase": "downloading", "state": "started", "detail": None,
                                              "done": 3, "total": 3, "updated_at": 5.0}

    def test_a_video_that_fails_is_recorded_and_the_rest_go_on(self, tmp_path):
        queue, job = _job(tmp_path)

        code = fetch(job.id, queue, FakeDownloader(failing={"video000002"}))

        assert code == 0
        manifest = json.loads((tmp_path / youtube.MANIFEST).read_text())
        assert manifest["errors"] == {"Lecture 2 [video000002].mp4": "Video unavailable"}
        assert queue.get(job.id).status is JobStatus.RUNNING

    def test_a_video_already_downloaded_is_not_fetched_again(self, tmp_path):
        queue, job = _job(tmp_path)
        (tmp_path / VIDEOS[0].file_name).write_bytes(b"mp4")
        downloader = FakeDownloader()

        fetch(job.id, queue, downloader)

        assert downloader.calls == ["video000002", "video000003"]

    def test_nothing_downloaded_fails_the_job_with_youtubes_reason(self, tmp_path):
        queue, job = _job(tmp_path)

        code = fetch(job.id, queue, FakeDownloader(failing={v.id for v in VIDEOS}))

        assert code == 1
        failed = queue.get(job.id)
        assert failed.status is JobStatus.FAILED
        assert failed.error == "None of the 3 videos could be downloaded: Video unavailable"

    def test_a_stopped_download_leaves_no_stream_fragment_for_ingestion_to_take_as_a_video(self, tmp_path):
        queue, job = _job(tmp_path, videos=VIDEOS[:1])
        for leftover in ("Lecture 2 [video000002].f137.mp4", "Lecture 2 [video000002].f140.m4a",
                         "Lecture 2 [video000002].mp4.part", "Lecture 2 [video000002].mp4.ytdl"):
            (tmp_path / leftover).write_bytes(b"x")

        fetch(job.id, queue, FakeDownloader())

        assert sorted(p.name for p in tmp_path.iterdir()) == [youtube.MANIFEST, VIDEOS[0].file_name]


class TestJsRuntimes:
    def test_deno_beside_this_python_is_used_by_its_path(self, monkeypatch, tmp_path):
        (tmp_path / "deno").write_bytes(b"")
        monkeypatch.setattr("sys.executable", str(tmp_path / "python"))
        assert _js_runtimes() == {"deno": {"path": str(tmp_path / "deno")}}

    def test_otherwise_whatever_runtime_is_on_the_path(self, monkeypatch, tmp_path):
        monkeypatch.setattr("sys.executable", str(tmp_path / "python"))
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/node" if name == "node" else None)
        assert _js_runtimes() == {"node": {}}

    def test_without_one_yt_dlp_keeps_its_default(self, monkeypatch, tmp_path):
        monkeypatch.setattr("sys.executable", str(tmp_path / "python"))
        monkeypatch.setattr("shutil.which", lambda name: None)
        assert _js_runtimes() == {"deno": {}}


class TestReportErrors:
    def test_videos_that_could_not_be_downloaded_are_listed_with_the_ingestion_failures(self, tmp_path):
        from web_deploy.ingest_worker.run_job import _download_errors
        youtube.write_manifest(tmp_path, VIDEOS)
        youtube.record_errors(tmp_path, {VIDEOS[1].file_name: "Private video"})

        assert _download_errors(str(tmp_path)) == {"Lecture 2 [video000002].mp4": "Private video"}

    def test_an_uploaded_batch_has_none(self, tmp_path):
        from web_deploy.ingest_worker.run_job import _download_errors
        assert _download_errors(str(tmp_path)) == {}
