from pathlib import Path

from web_deploy.ingest_worker.queue import IngestMode, IngestQueue, JobOrigin, JobStatus
from web_deploy.ingest_worker.run_job import ProgressWriter
from web_deploy.ingest_worker.supervisor import Supervisor, child_environment, firm_environment


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


class TestFirmGlossary:
    def _glossary(self, tenants, language, *names):
        folder = tenants / "legal-demo" / "glossaries" / language
        folder.mkdir(parents=True)
        for name in names:
            (folder / name).write_text("{}")

    def test_the_firms_glossary_for_the_language_is_what_the_child_reads(self, tmp_path):
        self._glossary(tmp_path, "en", "legal_disputes.json", "meta.json")

        env = firm_environment("legal-demo", "en", tmp_path, {"APPLIANCE_INDUSTRY": "FINANCE"})

        assert env["GLOSSARY_PATH"] == str(tmp_path / "legal-demo" / "glossaries")
        assert env["APPLIANCE_INDUSTRY"] == "LEGAL_DISPUTES"

    def test_without_one_the_shared_general_glossary_whatever_the_base_says(self, tmp_path):
        self._glossary(tmp_path, "en", "legal_disputes.json", "meta.json")

        env = firm_environment("legal-demo", "sr", tmp_path, {"APPLIANCE_INDUSTRY": "FINANCE", "GLOSSARY_PATH": "/x"})

        assert "APPLIANCE_INDUSTRY" not in env and "GLOSSARY_PATH" not in env

    def test_an_ambiguous_folder_is_not_guessed_at(self, tmp_path):
        self._glossary(tmp_path, "en", "a.json", "b.json")

        assert "APPLIANCE_INDUSTRY" not in firm_environment("legal-demo", "en", tmp_path, {})

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


class _Stager:
    """Records the staging calls around a job, as the real one would make them over rsync."""

    def __init__(self, events, fail_in=False, pending=False):
        self.events = events
        self.fail_in = fail_in
        self.pending = pending
        self.enabled = True

    def stage_in(self, job_id, firm, batch_dir):
        self.events.append(("in", firm, batch_dir))
        if self.fail_in:
            raise RuntimeError("rsync failed (255): connection refused")
        return []

    def stage_out(self, job_id):
        self.events.append(("out", job_id))

    def youtube_cookies_back(self, job_id, refused):
        self.events.append(("cookies_back", refused))

    def drop_youtube_cookies(self):
        self.events.append(("drop_cookies",))

    def has_pending(self):
        return self.pending

    def recover(self):
        self.events.append(("recover",))
        self.pending = False
        return 0


class TestStaging:
    def test_a_jobs_files_come_in_before_it_runs_and_its_results_go_back_after(self):
        queue = _queue()
        job = queue.enqueue("legal-demo", "u1", IngestMode.AUDIO, "en", "/t/legal-demo/uploads/u1/b", 1)
        events = []

        def spawn(claimed, env):
            events.append(("ingest",))
            queue.finish(claimed.id, {})
            return 0

        Supervisor(queue, Path("/t"), spawn=spawn, base_env={}, arbiter=_Arbiter(events),
                   stager=_Stager(events)).run_once()

        assert events == [("in", "legal-demo", "/t/legal-demo/uploads/u1/b"), "gpu", ("ingest",), ("out", job.id)]

    def test_a_failed_job_still_pushes_what_it_wrote(self):
        """Its rows are in the VPS database already; their files must not exist only on the box."""
        queue = _queue()
        job = queue.enqueue("legal-demo", "u1", IngestMode.AUDIO, "en", "/b", 1)
        events = []

        Supervisor(queue, Path("/t"), spawn=lambda j, e: -9, base_env={}, arbiter=_Arbiter([]),
                   stager=_Stager(events)).run_once()

        assert events[-1] == ("out", job.id)
        assert queue.get(job.id).status is JobStatus.FAILED

    def test_files_that_cannot_be_fetched_fail_the_job_without_the_gpu(self):
        queue = _queue()
        job = queue.enqueue("legal-demo", "u1", IngestMode.AUDIO, "en", "/b", 1)
        events = []

        Supervisor(queue, Path("/t"), spawn=lambda j, e: events.append("ingest") or 0, base_env={},
                   arbiter=_Arbiter(events), stager=_Stager(events, fail_in=True)).run_once()

        assert "gpu" not in events and "ingest" not in events
        assert "could not be fetched" in queue.get(job.id).error

    def test_a_stop_mid_job_leaves_the_results_for_the_next_start_to_push(self):
        queue = _queue()
        queue.enqueue("legal-demo", "u1", IngestMode.AUDIO, "en", "/b", 1)
        events = []
        supervisor = None

        def spawn(claimed, env):
            supervisor.stop()
            return -15

        supervisor = Supervisor(queue, Path("/t"), spawn=spawn, base_env={}, arbiter=_Arbiter([]),
                                stager=_Stager(events))
        supervisor.run_once()

        assert [e[0] for e in events] == ["in"]

    def test_a_youtube_batch_is_not_pulled_and_gets_its_videos_from_the_job_row(self, tmp_path):
        from web_deploy.common import youtube
        queue = _queue()
        folder = tmp_path / "legal-demo" / "uploads" / "u1" / "20261012-120000-abcdef"
        manifest = youtube.new_manifest([youtube.Video("video000001", "Lecture 1")])
        queue.enqueue("legal-demo", "u1", IngestMode.VIDEO, "en", str(folder), 1, origin=JobOrigin.YOUTUBE,
                      manifest=manifest)
        events = []
        seen = []

        def fetch(claimed, env):
            seen.append([v.id for v in youtube.pending(folder)])
            queue.fail(claimed.id, "stop here")
            return 1

        Supervisor(queue, tmp_path, spawn=lambda j, e: 0, fetch=fetch, base_env={}, arbiter=_Arbiter([]),
                   stager=_Stager(events)).run_once()

        assert events[0] == ("in", "legal-demo", None)
        assert seen == [["video000001"]]
        assert ("drop_cookies",) in events  # the YouTube account's cookies go, even though the download failed
        assert ("cookies_back", False) in events
        assert events.index(("cookies_back", False)) < events.index(("drop_cookies",))
        assert events.index(("drop_cookies",)) < next(i for i, e in enumerate(events) if e[0] == "out")


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

    def test_finished_jobs_folders_are_the_ones_cleanup_may_take(self):
        queue = _queue()
        done = queue.enqueue("legal-demo", "u1", IngestMode.AUDIO, "en", "/done", 1)
        queue.enqueue("legal-demo", "u1", IngestMode.AUDIO, "en", "/queued", 1)
        cancelled = queue.enqueue("legal-demo", "u1", IngestMode.AUDIO, "en", "/cancelled", 1)
        queue.claim_next()
        queue.finish(done.id, {})
        queue.cancel(cancelled.id, "legal-demo")

        assert queue.finished_batch_dirs() == {"/done", "/cancelled"}

