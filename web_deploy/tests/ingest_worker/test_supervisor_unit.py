from pathlib import Path

from web_deploy.ingest_worker.queue import IngestMode, IngestQueue, JobOrigin, JobStatus
from web_deploy.ingest_worker.run_job import ProgressWriter
from web_deploy.ingest_worker.supervisor import Supervisor, child_environment


def _queue():
    return IngestQueue.connect("sqlite://")


class TestChildEnvironment:
    def test_a_job_runs_in_its_firms_database_and_vault_in_its_batchs_language_on_the_ingestion_half(self):
        queue = _queue()
        job = queue.enqueue("legal-demo", "u1", IngestMode.VIDEO, "sr", "/b", 1)

        env = child_environment(job, Path("/t"), {"PATH": "/bin", "INGEST_VRAM_HEADROOM_MB": "2000",
                                                 "ENGINE_CONCURRENT_LIVE": "true", "LLM_BACKEND": "llama_server"})

        assert env["PATH"] == "/bin"
        assert env["POSTGRES_DB"] == "onyx_firm_legal_demo"
        assert env["ONYX_STORAGE_PATH"] == "/t/legal-demo/vault"
        assert env["REPORTS_PATH"] == "/t/legal-demo/reports"
        assert env["APPLIANCE_LANGUAGE"] == "sr"
        assert (env["LLM_BACKEND"], env["ENGINE_CONCURRENT_LIVE"], env["MODEL_KEEP_WARM"]) == ("llama_cpp", "false", "false")
        assert (env["MODEL_PINNED_TYPES"], env["MODEL_EXTERNAL_TYPES"]) == ("[]", "[]")
        assert env["VRAM_HEADROOM_MB"] == "2000"


class TestSupervisor:
    def test_runs_the_oldest_job_in_a_child(self):
        queue = _queue()
        job = queue.enqueue("legal-demo", "u1", IngestMode.AUDIO, "en", "/b", 1)
        seen = []

        def spawn(claimed, env):
            seen.append((claimed.id, env["POSTGRES_DB"]))
            queue.finish(claimed.id, {"succeeded": 1})
            return 0

        supervisor = Supervisor(queue, Path("/t"), spawn=spawn, base_env={})
        assert supervisor.run_once() is True
        assert supervisor.run_once() is False
        assert seen == [(job.id, "onyx_firm_legal_demo")]
        assert queue.get(job.id).status is JobStatus.DONE

    def test_a_child_that_dies_without_a_verdict_fails_the_job(self):
        queue = _queue()
        job = queue.enqueue("legal-demo", "u1", IngestMode.AUDIO, "en", "/b", 1)
        Supervisor(queue, Path("/t"), spawn=lambda j, e: -9, base_env={}).run_once()
        failed = queue.get(job.id)
        assert failed.status is JobStatus.FAILED
        assert "exit code -9" in failed.error

    def test_a_child_that_cannot_start_fails_the_job(self):
        queue = _queue()
        job = queue.enqueue("legal-demo", "u1", IngestMode.AUDIO, "en", "/b", 1)

        def boom(j, e):
            raise OSError("no python")

        Supervisor(queue, Path("/t"), spawn=boom, base_env={}).run_once()
        assert "no python" in queue.get(job.id).error


class TestProgressWriter:
    def test_progress_is_throttled_but_counts_every_step(self):
        queue = _queue()
        job = queue.enqueue("legal-demo", "u1", IngestMode.AUDIO, "en", "/b", 1)
        now = [0.0]
        writer = ProgressWriter(queue, job.id, clock=lambda: now[0], wall=lambda: 1000.0)

        writer("transcribe", "started", None)
        writer("transcribe", "started", None)  # within the interval: skipped
        writer("transcribe", "completed", None, "done")

        assert queue.get(job.id).progress == {"phase": "transcribe", "state": "completed", "detail": "done",
                                              "steps_completed": 1, "steps_failed": 0,
                                              "done_by_phase": {"transcribe": 1}, "files": 0,
                                              "updated_at": 1000.0}

    def test_counts_work_per_job_and_distinct_files_for_the_step_bars(self):
        queue = _queue()
        job = queue.enqueue("legal-demo", "u1", IngestMode.AUDIO, "en", "/b", 2)
        now = [0.0]
        writer = ProgressWriter(queue, job.id, clock=lambda: now[0], wall=lambda: 1000.0)

        writer("vad", "completed", "a")
        writer("vad", "failed", "b")
        writer("vad", "completed", "a")
        writer("transcribe", "started", "a")  # a new job is written at once, inside the interval

        progress = queue.get(job.id).progress
        assert progress["phase"] == "transcribe"
        assert progress["done_by_phase"] == {"vad": 3, "transcribe": 0}
        assert progress["files"] == 2
        assert (progress["steps_completed"], progress["steps_failed"]) == (2, 1)


class TestJobTimes:
    def test_times_come_back_in_utc_from_sqlite_which_drops_the_zone(self):
        queue = _queue()
        job = queue.enqueue("legal-demo", "u1", IngestMode.AUDIO, "en", "/b", 1)

        created = queue.get(job.id).public()["created_at"]

        assert created.endswith("+00:00")


class _Arbiter:
    """Records when the GPU was asked for, relative to the download."""

    def __init__(self, events):
        self.events = events

    def before_job(self):
        self.events.append("gpu")

    def when_idle(self):
        pass

    def stop(self):
        pass


class TestYouTubeBatches:
    @staticmethod
    def _youtube_job(queue, folder, videos=2):
        from web_deploy.common import youtube
        youtube.write_manifest(folder, [youtube.Video(f"video{i:06d}", f"Lecture {i}") for i in range(1, videos + 1)])
        return queue.enqueue("legal-demo", "u1", IngestMode.VIDEO, "en", str(folder), videos, origin=JobOrigin.YOUTUBE)

    def test_videos_are_downloaded_before_the_gpu_is_asked_for(self, tmp_path):
        queue = _queue()
        job = self._youtube_job(queue, tmp_path)
        events = []

        def fetch(claimed, env):
            events.append("download")
            return 0

        def spawn(claimed, env):
            events.append("ingest")
            queue.finish(claimed.id, {})
            return 0

        Supervisor(queue, Path("/t"), spawn=spawn, fetch=fetch, base_env={}, arbiter=_Arbiter(events)).run_once()

        assert events == ["download", "gpu", "ingest"]
        assert queue.get(job.id).status is JobStatus.DONE

    def test_a_batch_already_downloaded_goes_straight_to_ingestion(self, tmp_path):
        from web_deploy.common import youtube
        queue = _queue()
        job = self._youtube_job(queue, tmp_path, videos=1)
        (tmp_path / youtube.Video("video000001", "Lecture 1").file_name).write_bytes(b"mp4")
        events = []

        Supervisor(queue, Path("/t"), spawn=lambda j, e: events.append("ingest") or queue.finish(j.id, {}) or 0,
                   fetch=lambda j, e: events.append("download") or 0, base_env={}, arbiter=_Arbiter(events)).run_once()

        assert events == ["gpu", "ingest"]
        assert queue.get(job.id).status is JobStatus.DONE

    def test_a_download_that_got_nothing_fails_without_taking_the_gpu(self, tmp_path):
        queue = _queue()
        job = self._youtube_job(queue, tmp_path)
        events = []

        def fetch(claimed, env):
            queue.fail(claimed.id, "None of the 2 videos could be downloaded: Private video")
            return 1

        Supervisor(queue, Path("/t"), spawn=lambda j, e: events.append("ingest") or 0, fetch=fetch, base_env={},
                   arbiter=_Arbiter(events)).run_once()

        assert events == []
        assert queue.get(job.id).error.startswith("None of the 2 videos")

    def test_a_download_that_crashes_fails_the_job(self, tmp_path):
        queue = _queue()
        job = self._youtube_job(queue, tmp_path)

        Supervisor(queue, Path("/t"), spawn=lambda j, e: 0, fetch=lambda j, e: -9, base_env={},
                   arbiter=_Arbiter([])).run_once()

        assert "download stopped unexpectedly (exit code -9)" in queue.get(job.id).error

    def test_a_stop_during_the_download_leaves_the_job_to_be_requeued(self, tmp_path):
        queue = _queue()
        job = self._youtube_job(queue, tmp_path)
        events = []
        supervisor = None

        def fetch(claimed, env):
            supervisor.stop()
            return -15

        supervisor = Supervisor(queue, Path("/t"), spawn=lambda j, e: events.append("ingest") or 0, fetch=fetch,
                                base_env={}, arbiter=_Arbiter(events))
        supervisor.run_once()

        assert events == []
        assert queue.get(job.id).status is JobStatus.RUNNING
        assert queue.recover_interrupted() == 1

    def test_an_uploaded_video_batch_is_never_downloaded(self):
        queue = _queue()
        queue.enqueue("legal-demo", "u1", IngestMode.VIDEO, "en", "/b", 1)
        fetched = []

        Supervisor(queue, Path("/t"), spawn=lambda j, e: queue.finish(j.id, {}) or 0,
                   fetch=lambda j, e: fetched.append(j) or 0, base_env={}, arbiter=_Arbiter([])).run_once()

        assert fetched == []


class TestUploadCleanup:
    DAY = 24 * 3600

    @staticmethod
    def _batch(tenants, firm="legal-demo", owner="u1", batch="20261009-120000-abcdef"):
        folder = tenants / firm / "uploads" / owner / batch
        folder.mkdir(parents=True)
        (folder / "talk.mp4").write_bytes(b"video")
        return folder

    @staticmethod
    def _age(folder, seconds_ago, now):
        import os
        for path in [folder, *folder.rglob("*")]:
            os.utime(path, (now - seconds_ago, now - seconds_ago))

    @staticmethod
    def _supervisor(queue, tenants, spawn, now=None, **kwargs):
        return Supervisor(queue, tenants, spawn=spawn, base_env={}, arbiter=_Arbiter([]),
                          clock=(lambda: now) if now is not None else __import__("time").time, **kwargs)

    def test_a_finished_jobs_uploads_are_removed(self, tmp_path):
        queue = _queue()
        folder = self._batch(tmp_path)
        job = queue.enqueue("legal-demo", "u1", IngestMode.VIDEO, "en", str(folder), 1)

        def spawn(claimed, env):
            queue.finish(claimed.id, {"succeeded": 1})
            return 0

        self._supervisor(queue, tmp_path, spawn).run_once()

        assert queue.get(job.id).status is JobStatus.DONE
        assert not folder.exists()
        assert (tmp_path / "legal-demo" / "uploads" / "u1").is_dir()

    def test_a_failed_jobs_uploads_are_removed_too(self, tmp_path):
        queue = _queue()
        folder = self._batch(tmp_path)
        queue.enqueue("legal-demo", "u1", IngestMode.VIDEO, "en", str(folder), 1)

        self._supervisor(queue, tmp_path, lambda j, e: -9).run_once()

        assert not folder.exists()

    def test_a_job_interrupted_by_a_stop_keeps_its_uploads_for_the_retry(self, tmp_path):
        queue = _queue()
        folder = self._batch(tmp_path)
        queue.enqueue("legal-demo", "u1", IngestMode.VIDEO, "en", str(folder), 1)
        supervisor = None

        def spawn(claimed, env):
            supervisor.stop()  # the box is stopping: the child is terminated mid-job
            return -15

        supervisor = self._supervisor(queue, tmp_path, spawn)
        supervisor.run_once()

        assert folder.exists()

    def test_nothing_outside_the_firms_uploads_is_ever_removed(self, tmp_path):
        queue = _queue()
        elsewhere = tmp_path / "samples"
        elsewhere.mkdir()
        (elsewhere / "keep.mp4").write_bytes(b"x")
        queue.enqueue("legal-demo", "u1", IngestMode.VIDEO, "en", str(elsewhere), 1)

        def spawn(claimed, env):
            queue.finish(claimed.id, {})
            return 0

        self._supervisor(queue, tmp_path / "tenants", spawn).run_once()

        assert (elsewhere / "keep.mp4").exists()

    def test_the_sweep_removes_only_old_folders_no_job_will_read(self, tmp_path):
        now = 2_000_000_000.0
        queue = _queue()
        abandoned = self._batch(tmp_path, batch="20261001-000000-aaaaaa")
        uploading = self._batch(tmp_path, batch="20261009-000000-bbbbbb")
        waiting = self._batch(tmp_path, firm="clinic-demo", batch="20261002-000000-cccccc")
        self._age(abandoned, 2 * self.DAY, now)
        self._age(uploading, 2 * self.DAY, now)
        (uploading / ".big.mp4.part").write_bytes(b"still arriving")
        import os
        os.utime(uploading / ".big.mp4.part", (now - 60, now - 60))
        self._age(waiting, 3 * self.DAY, now)
        queue.enqueue("clinic-demo", "u1", IngestMode.AUDIO, "en", str(waiting), 1)

        removed = self._supervisor(queue, tmp_path, lambda j, e: 0, now=now).sweep_uploads()

        assert removed == 1
        assert not abandoned.exists()
        assert uploading.exists() and waiting.exists()

    def test_an_idle_supervisor_sweeps_at_most_once_an_hour(self, tmp_path):
        now = [2_000_000_000.0]
        queue = _queue()
        supervisor = Supervisor(queue, tmp_path, spawn=lambda j, e: 0, base_env={}, arbiter=_Arbiter([]),
                                clock=lambda: now[0])
        first = self._batch(tmp_path, batch="20261001-000000-aaaaaa")
        self._age(first, 2 * self.DAY, now[0])

        supervisor.run_once()
        assert not first.exists()

        second = self._batch(tmp_path, batch="20261001-000000-dddddd")
        self._age(second, 2 * self.DAY, now[0])
        now[0] += 60
        supervisor.run_once()
        assert second.exists()

        now[0] += 3600
        supervisor.run_once()
        assert not second.exists()


class TestActiveBatchDirs:
    def test_only_queued_and_running_jobs_count(self):
        queue = _queue()
        queue.enqueue("legal-demo", "u1", IngestMode.AUDIO, "en", "/running", 1)
        finished = queue.enqueue("legal-demo", "u1", IngestMode.AUDIO, "en", "/finished", 1)
        queue.enqueue("legal-demo", "u1", IngestMode.AUDIO, "en", "/queued", 1)
        cancelled = queue.enqueue("legal-demo", "u1", IngestMode.AUDIO, "en", "/cancelled", 1)
        queue.claim_next()
        queue.claim_next()
        queue.finish(finished.id, {})
        queue.cancel(cancelled.id, "legal-demo")

        assert queue.active_batch_dirs() == {"/running", "/queued"}
